"""Local storage: one SQLite file plus JPEG snapshots of item sightings.

Schema
------
people          one row per person who agreed to be remembered
face_exemplars  that person's face embeddings (512 float32), captured over several seconds
items           personal items ("keys", "wallet")
item_exemplars  image embeddings of each item from several angles
sightings       where/when each item was last seen (+ snapshot path, context phrase,
                named place, nearby landmark, who you were with, Wi-Fi network)
places          rooms/places the user named ("this is the kitchen"): scene embeddings
                plus the Wi-Fi access points and Bluetooth devices seen there

What is NOT stored: face images of anyone, and anything about unknown faces.
Unknown faces exist only in memory while they're being tracked.

Embeddings are raw float32 bytes. All methods take one lock around a single
connection, because several threads use the store.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    consent_at   REAL,
    last_seen_at REAL
);
CREATE TABLE IF NOT EXISTS face_exemplars (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    quality    REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_face_person ON face_exemplars(person_id);

CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    key        TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS item_exemplars (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_item_ex ON item_exemplars(item_id);

CREATE TABLE IF NOT EXISTS sightings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id       INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    ts            REAL NOT NULL,
    first_ts      REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    frame_w       INTEGER, frame_h INTEGER,
    score         REAL,
    snapshot_path TEXT,
    context       TEXT NOT NULL DEFAULT '',
    camera_id     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sightings_item ON sightings(item_id, ts);

CREATE TABLE IF NOT EXISTS places (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    key         TEXT NOT NULL UNIQUE,
    created_at  REAL NOT NULL,
    wifi_ssid   TEXT NOT NULL DEFAULT '',   -- network name(s) seen there, '|'-separated
    wifi_bssids TEXT NOT NULL DEFAULT '[]',
    ble         TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS place_views (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id   INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""

# Columns added after the first release; added to older databases on open.
_SIGHTING_COLUMNS = {
    "place": "TEXT NOT NULL DEFAULT ''",        # named place ("kitchen") if recognised
    "companions": "TEXT NOT NULL DEFAULT ''",   # known people in view at the time ("Sarah")
    "wifi_ssid": "TEXT NOT NULL DEFAULT ''",    # Wi-Fi network the laptop was on
    "near": "TEXT NOT NULL DEFAULT ''",         # landmark seen in or just around that frame ("bed")
}
_PLACE_COLUMNS = {
    "mine": "INTEGER NOT NULL DEFAULT 0",       # "this is MY bedroom" -> "your bedroom"; else "the library"
    "landmarks": "TEXT NOT NULL DEFAULT '{}'",  # landmarks seen while it was learned: {"bed": 7, "desk": 4}
}

_ALIASES = {
    "airpod": "airpods", "air pods": "airpods", "air pod": "airpods", "earbuds": "airpods",
    "earbud": "airpods", "ear buds": "airpods", "earbuds case": "airpods", "airpods case": "airpods",
    "key": "keys", "car keys": "keys", "house keys": "keys", "keychain": "keys",
    "headphone": "headphones", "head phones": "headphones",
    "spectacles": "glasses", "sunglasses": "sunglasses",
    "cell phone": "phone", "cellphone": "phone", "mobile": "phone", "iphone": "phone",
}
# Words that are already "singular" for our purposes (don't strip the s).
_NO_SINGULAR = {"glasses", "sunglasses", "headphones", "earphones", "keys", "airpods", "scissors", "pants"}


def item_key(name: str) -> str:
    """Canonical key so "my keys", "Keys" and "the key" all find the same item."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"^(my|the|a|an|our|your)\s+", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = _ALIASES.get(s, s)
    if s not in _NO_SINGULAR and s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]
    return s.replace(" ", "")


def name_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _blob(e: np.ndarray) -> bytes:
    return np.asarray(e, dtype=np.float32).tobytes()


def _unblob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


@dataclass
class Sighting:
    id: int
    item_id: int
    item: str
    ts: float
    first_ts: float
    bbox: Tuple[float, float, float, float]
    frame_size: Tuple[int, int]
    score: float
    snapshot_path: Optional[str]
    context: str
    camera_id: str
    place: str = ""
    companions: str = ""
    wifi_ssid: str = ""
    near: str = ""


class Storage:
    def __init__(self, db_path: str, snapshot_dir: str = "data/snapshots"):
        self.path = str(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            for table, columns in (("sightings", _SIGHTING_COLUMNS), ("places", _PLACE_COLUMNS)):
                have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
                for col, decl in columns.items():
                    if col not in have:
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _q(self, sql: str, params: Sequence = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _x(self, sql: str, params: Sequence = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    # ================================================================ people
    def add_person(self, name: str, embeddings: Sequence[np.ndarray], qualities: Sequence[float],
                   now: Optional[float] = None) -> int:
        """Create a person (or add views to an existing one with the same name)."""
        now = time.time() if now is None else now
        existing = self.find_person(name)
        with self._lock:
            if existing is not None:
                pid = existing["id"]
                self._conn.execute("UPDATE people SET consent_at = ? WHERE id = ?", (now, pid))
            else:
                cur = self._conn.execute(
                    "INSERT INTO people (name, created_at, consent_at) VALUES (?, ?, ?)", (name, now, now)
                )
                pid = int(cur.lastrowid)
            for e, q in zip(embeddings, qualities):
                self._conn.execute(
                    "INSERT INTO face_exemplars (person_id, embedding, quality, created_at) VALUES (?, ?, ?, ?)",
                    (pid, _blob(e), float(q), now),
                )
            self._conn.commit()
        return pid

    def find_person(self, name: str) -> Optional[Dict]:
        key = name_key(name)
        if not key:
            return None
        for row in self._q("SELECT id, name, created_at, last_seen_at FROM people"):
            if name_key(row["name"]) == key:
                return dict(row)
        return None

    def get_person(self, person_id: int) -> Optional[Dict]:
        rows = self._q("SELECT id, name, created_at, last_seen_at FROM people WHERE id = ?", (person_id,))
        return dict(rows[0]) if rows else None

    def list_people(self) -> List[Dict]:
        rows = self._q(
            "SELECT p.id, p.name, p.created_at, p.last_seen_at, "
            "(SELECT COUNT(*) FROM face_exemplars f WHERE f.person_id = p.id) AS n_exemplars "
            "FROM people p ORDER BY p.name COLLATE NOCASE"
        )
        return [dict(r) for r in rows]

    def rename_person(self, person_id: int, new_name: str) -> None:
        self._x("UPDATE people SET name = ? WHERE id = ?", (new_name, person_id))

    def delete_all_people(self) -> int:
        n = len(self._q("SELECT id FROM people"))
        self._x("DELETE FROM people")
        return n

    def delete_person(self, person_id: int) -> None:
        self._x("DELETE FROM people WHERE id = ?", (person_id,))

    def touch_person(self, person_id: int, now: float) -> None:
        self._x("UPDATE people SET last_seen_at = ? WHERE id = ?", (now, person_id))

    def face_gallery(self) -> Iterator[Tuple[int, str, np.ndarray, List[int]]]:
        """(person_id, name, embeddings (k, d), exemplar_ids) for everyone."""
        for p in self._q("SELECT id, name FROM people"):
            rows = self._q(
                "SELECT id, embedding FROM face_exemplars WHERE person_id = ? ORDER BY id", (p["id"],)
            )
            if rows:
                yield p["id"], p["name"], np.stack([_unblob(r["embedding"]) for r in rows]), [r["id"] for r in rows]

    # ================================================================ items
    def find_item(self, name: str) -> Optional[Dict]:
        rows = self._q("SELECT id, name, key, created_at FROM items WHERE key = ?", (item_key(name),))
        return dict(rows[0]) if rows else None

    def rename_item(self, item_id: int, new_name: str) -> str:
        """Rename an item; its views and sightings stay. Returns the stored name."""
        clean = re.sub(r"^(my|the|a|an|our)\s+", "", new_name.strip(), flags=re.I).lower()
        self._x("UPDATE items SET name = ?, key = ? WHERE id = ?", (clean, item_key(clean), item_id))
        return clean

    def get_item(self, item_id: int) -> Optional[Dict]:
        rows = self._q("SELECT id, name, key, created_at FROM items WHERE id = ?", (item_id,))
        return dict(rows[0]) if rows else None

    def add_item(self, name: str, embeddings: Sequence[np.ndarray], now: Optional[float] = None) -> int:
        """Create an item (or add views to an existing item with the same name)."""
        now = time.time() if now is None else now
        clean = re.sub(r"^(my|the|a|an)\s+", "", name.strip(), flags=re.I)
        existing = self.find_item(clean)
        with self._lock:
            if existing:
                iid = existing["id"]
            else:
                cur = self._conn.execute(
                    "INSERT INTO items (name, key, created_at) VALUES (?, ?, ?)", (clean.lower(), item_key(clean), now)
                )
                iid = int(cur.lastrowid)
            for e in embeddings:
                self._conn.execute(
                    "INSERT INTO item_exemplars (item_id, embedding, created_at) VALUES (?, ?, ?)",
                    (iid, _blob(e), now),
                )
            self._conn.commit()
        return iid

    def list_items(self) -> List[Dict]:
        rows = self._q(
            "SELECT i.id, i.name, i.key, i.created_at, "
            "(SELECT COUNT(*) FROM item_exemplars e WHERE e.item_id = i.id) AS n_exemplars "
            "FROM items i ORDER BY i.name"
        )
        return [dict(r) for r in rows]

    def delete_item(self, item_id: int) -> int:
        """Delete an item, its views, sightings and snapshot files. Returns files removed."""
        paths = [r["snapshot_path"] for r in self._q("SELECT snapshot_path FROM sightings WHERE item_id = ?", (item_id,))]
        self._x("DELETE FROM items WHERE id = ?", (item_id,))
        return self._remove_files(paths)

    def delete_all_items(self) -> int:
        n = 0
        for it in self.list_items():
            self.delete_item(it["id"])
            n += 1
        return n

    def item_gallery(self) -> Iterator[Tuple[int, str, np.ndarray]]:
        for it in self._q("SELECT id, name FROM items"):
            rows = self._q("SELECT embedding FROM item_exemplars WHERE item_id = ? ORDER BY id", (it["id"],))
            if rows:
                yield it["id"], it["name"], np.stack([_unblob(r["embedding"]) for r in rows])

    # ================================================================ places
    def add_place(self, name: str, embeddings: Sequence[np.ndarray], wifi_ssid: str = "",
                  wifi_bssids: Sequence[str] = (), ble: Optional[Dict[str, float]] = None,
                  now: Optional[float] = None, mine: Optional[bool] = None,
                  landmarks: Optional[Dict[str, int]] = None) -> int:
        """Create a named place (or add views/radio info to an existing one).
        mine: the user said "my bedroom" (True) or "the library" (False); None = unchanged."""
        now = time.time() if now is None else now
        clean = re.sub(r"^(my|the|a|an|our)\s+", "", name.strip(), flags=re.I).lower()
        existing = self.find_place(clean)
        with self._lock:
            if existing:
                pid = existing["id"]
                bssids = sorted(set(existing["wifi_bssids"]) | set(wifi_bssids))
                ble_all = dict(existing["ble"])
                ble_all.update(ble or {})
                ssids = "|".join(sorted({x for x in existing["wifi_ssid"].split("|") + [wifi_ssid] if x}))
                self._conn.execute("UPDATE places SET wifi_ssid = ?, wifi_bssids = ?, ble = ? WHERE id = ?",
                                   (ssids, json.dumps(bssids), json.dumps(ble_all), pid))
                if mine is not None:
                    self._conn.execute("UPDATE places SET mine = ? WHERE id = ?", (int(bool(mine)), pid))
                if landmarks:
                    merged = dict(existing["landmarks"])
                    for k, n in landmarks.items():
                        merged[k] = merged.get(k, 0) + int(n)
                    self._conn.execute("UPDATE places SET landmarks = ? WHERE id = ?", (json.dumps(merged), pid))
            else:
                cur = self._conn.execute(
                    "INSERT INTO places (name, key, created_at, wifi_ssid, wifi_bssids, ble, mine, landmarks) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (clean, item_key(clean), now, wifi_ssid, json.dumps(sorted(set(wifi_bssids))),
                     json.dumps(ble or {}), int(bool(mine)), json.dumps(dict(landmarks or {}))))
                pid = int(cur.lastrowid)
            for e in embeddings:
                self._conn.execute("INSERT INTO place_views (place_id, embedding, created_at) VALUES (?, ?, ?)",
                                   (pid, _blob(e), now))
            self._conn.commit()
        return pid

    def _place_dict(self, r) -> Dict:
        d = dict(r)
        d["wifi_bssids"] = json.loads(d.get("wifi_bssids") or "[]")
        d["ble"] = json.loads(d.get("ble") or "{}")
        d["mine"] = bool(d.get("mine", 0))
        d["landmarks"] = json.loads(d.get("landmarks") or "{}")
        return d

    def rename_place(self, place_id: int, new_name: str, mine: Optional[bool] = None) -> str:
        """Rename a place (and the sightings that mention it). Returns the stored name."""
        clean = re.sub(r"^(my|the|a|an|our)\s+", "", new_name.strip(), flags=re.I).lower()
        with self._lock:
            row = self._conn.execute("SELECT name FROM places WHERE id = ?", (place_id,)).fetchone()
            if row is None:
                return clean
            self._conn.execute("UPDATE places SET name = ?, key = ? WHERE id = ?", (clean, item_key(clean), place_id))
            if mine is not None:
                self._conn.execute("UPDATE places SET mine = ? WHERE id = ?", (int(bool(mine)), place_id))
            self._conn.execute("UPDATE sightings SET place = ? WHERE place = ?", (clean, row["name"]))
            self._conn.commit()
        return clean

    def place_label(self, name: str) -> str:
        """How to say a saved place: "your bedroom", "the library", "Sarah's room"."""
        if not name:
            return ""
        if "'s " in name or name.startswith(("my ", "the ")):
            return name
        p = self.find_place(name)
        return f"{'your' if p and p['mine'] else 'the'} {name}"

    def find_place(self, name: str) -> Optional[Dict]:
        rows = self._q("SELECT * FROM places WHERE key = ?", (item_key(name),))
        return self._place_dict(rows[0]) if rows else None

    def list_places(self) -> List[Dict]:
        return [self._place_dict(r) for r in self._q("SELECT * FROM places ORDER BY name")]

    def delete_place(self, place_id: int) -> None:
        self._x("DELETE FROM places WHERE id = ?", (place_id,))

    def delete_all_places(self) -> int:
        n = len(self._q("SELECT id FROM places"))
        self._x("DELETE FROM places")
        return n

    def place_gallery(self) -> Iterator[Tuple[Dict, np.ndarray]]:
        for p in self.list_places():
            rows = self._q("SELECT embedding FROM place_views WHERE place_id = ? ORDER BY id", (p["id"],))
            if rows:
                yield p, np.stack([_unblob(r["embedding"]) for r in rows])

    # ================================================================ sightings
    def add_sighting(self, item_id: int, ts: float, bbox: Sequence[float], frame_size: Tuple[int, int],
                     score: float, snapshot_path: Optional[str], context: str, camera_id: str,
                     keep: int = 20, place: str = "", companions: str = "", wifi_ssid: str = "",
                     near: str = "") -> int:
        cur = self._x(
            "INSERT INTO sightings (item_id, ts, first_ts, x1, y1, x2, y2, frame_w, frame_h, score, "
            "snapshot_path, context, camera_id, place, companions, wifi_ssid, near) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, ts, ts, *[float(v) for v in bbox], int(frame_size[0]), int(frame_size[1]),
             float(score), snapshot_path, context, camera_id, place, companions, wifi_ssid, near),
        )
        sid = int(cur.lastrowid)
        self._prune(item_id, keep)
        return sid

    def refresh_sighting(self, sighting_id: int, ts: float, bbox: Sequence[float], score: float,
                         context: Optional[str] = None, snapshot_path: Optional[str] = None,
                         place: Optional[str] = None, companions: Optional[str] = None) -> None:
        """Same place as the latest sighting: just move its timestamp (and box) forward."""
        sets = "ts = ?, x1 = ?, y1 = ?, x2 = ?, y2 = ?, score = MAX(score, ?)"
        params: list = [ts, *[float(v) for v in bbox], float(score)]
        old_snapshot = None
        for col, val in (("context", context), ("place", place), ("companions", companions)):
            if val is not None:
                sets += f", {col} = ?"
                params.append(val)
        if snapshot_path is not None:
            rows = self._q("SELECT snapshot_path FROM sightings WHERE id = ?", (sighting_id,))
            old_snapshot = rows[0]["snapshot_path"] if rows else None
            sets += ", snapshot_path = ?"
            params.append(snapshot_path)
        self._x(f"UPDATE sightings SET {sets} WHERE id = ?", (*params, sighting_id))
        if old_snapshot and old_snapshot != snapshot_path:
            self._remove_files([old_snapshot])

    def set_sighting_extras(self, sighting_id: int, near: Optional[str] = None, place: Optional[str] = None) -> None:
        """Fill in the room / nearby landmark found in the frames around a sighting."""
        sets, params = [], []
        for col, val in (("near", near), ("place", place)):
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)
        if sets:
            self._x(f"UPDATE sightings SET {', '.join(sets)} WHERE id = ?", (*params, sighting_id))

    def _prune(self, item_id: int, keep: int) -> None:
        rows = self._q("SELECT id, snapshot_path FROM sightings WHERE item_id = ? ORDER BY ts DESC", (item_id,))
        old = rows[keep:]
        if old:
            self._x(f"DELETE FROM sightings WHERE id IN ({','.join('?' * len(old))})", [r["id"] for r in old])
            self._remove_files([r["snapshot_path"] for r in old])

    def _row_to_sighting(self, r: sqlite3.Row) -> Sighting:
        return Sighting(
            id=r["id"], item_id=r["item_id"], item=r["name"], ts=r["ts"], first_ts=r["first_ts"],
            bbox=(r["x1"], r["y1"], r["x2"], r["y2"]), frame_size=(r["frame_w"], r["frame_h"]),
            score=r["score"], snapshot_path=r["snapshot_path"], context=r["context"], camera_id=r["camera_id"],
            place=r["place"], companions=r["companions"], wifi_ssid=r["wifi_ssid"], near=r["near"],
        )

    def last_sighting(self, item_id: int) -> Optional[Sighting]:
        rows = self._q(
            "SELECT s.*, i.name FROM sightings s JOIN items i ON i.id = s.item_id "
            "WHERE s.item_id = ? ORDER BY s.ts DESC LIMIT 1",
            (item_id,),
        )
        return self._row_to_sighting(rows[0]) if rows else None

    def sightings(self, item_id: int, limit: int = 20) -> List[Sighting]:
        rows = self._q(
            "SELECT s.*, i.name FROM sightings s JOIN items i ON i.id = s.item_id "
            "WHERE s.item_id = ? ORDER BY s.ts DESC LIMIT ?",
            (item_id, limit),
        )
        return [self._row_to_sighting(r) for r in rows]

    @staticmethod
    def _remove_files(paths) -> int:
        n = 0
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    n += 1
                except OSError:
                    pass
        return n
