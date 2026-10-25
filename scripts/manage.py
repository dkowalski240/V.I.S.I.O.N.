"""Look at and clean up what the assistant remembers (run while the assistant is stopped;
while it's running, use the voice/typed commands instead, e.g. "forget everyone").

    python scripts/manage.py list                 people, items, places, sightings
    python scripts/manage.py forget Sarah         one person, item or place
    python scripts/manage.py rename wallet "black wallet"   rename a person, item or place
    python scripts/manage.py forget-people        every enrolled person
    python scripts/manage.py forget-items         every item, its sightings and snapshots
    python scripts/manage.py forget-places        every named place
    python scripts/manage.py reset                all of the above
    python scripts/manage.py --db data/replay.db list
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from assistant.config import load_config  # noqa: E402
from assistant.geometry import time_ago_phrase  # noqa: E402
from assistant.storage import Storage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "forget", "rename", "forget-people", "forget-items",
                                        "forget-places", "reset"])
    ap.add_argument("name", nargs="?", help="for forget / rename: the person, item or place")
    ap.add_argument("new_name", nargs="?", help="for rename: the new name")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--db", help="database (default: storage.db_path from the config)")
    a = ap.parse_args()
    cfg = load_config(a.config if os.path.exists(a.config) else None)
    st = Storage(a.db or cfg.storage.db_path, cfg.storage.snapshot_dir)
    now = time.time()
    try:
        if a.command == "list":
            people = st.list_people()
            print(f"People ({len(people)}):")
            for p in people:
                seen = time_ago_phrase(now - p["last_seen_at"]) if p["last_seen_at"] else "never announced"
                print(f"  {p['name']:20s} {p['n_exemplars']:2d} views, last seen {seen}")
            items = st.list_items()
            print(f"Items ({len(items)}):")
            for it in items:
                s = st.last_sighting(it["id"])
                where = (f"last seen {time_ago_phrase(now - s.ts)}"
                         + (f" in {st.place_label(s.place)}" if s.place else "") + (f", {s.context}" if s.context else "")
                         if s else "never seen")
                print(f"  {it['name']:20s} {it['n_exemplars']:2d} views, {where}")
            places = st.list_places()
            print(f"Places ({len(places)}):")
            for pl in places:
                radio = []
                if pl["wifi_ssid"]:
                    radio.append(f"Wi-Fi {pl['wifi_ssid']} ({len(pl['wifi_bssids'])} access points)")
                if pl["ble"]:
                    radio.append(f"{len(pl['ble'])} Bluetooth devices")
                print(f"  {pl['name']:20s} {', '.join(radio) or 'camera views only'}")
        elif a.command == "forget":
            if not a.name:
                ap.error("forget needs a name")
            p, it, pl = st.find_person(a.name), st.find_item(a.name), st.find_place(a.name)
            if p:
                st.delete_person(p["id"])
                print(f"Forgot person {p['name']}")
            elif it:
                st.delete_item(it["id"])
                print(f"Forgot item {it['name']}")
            elif pl:
                st.delete_place(pl["id"])
                print(f"Forgot place {pl['name']}")
            else:
                print(f"Nothing called {a.name!r}")
                return 1
        elif a.command == "rename":
            if not (a.name and a.new_name):
                ap.error('rename needs the old and the new name, e.g. rename wallet "black wallet"')
            p, it, pl = st.find_person(a.name), st.find_item(a.name), st.find_place(a.name)
            if p:
                st.rename_person(p["id"], a.new_name)
                print(f"Renamed person {p['name']} to {a.new_name}")
            elif it:
                other = st.find_item(a.new_name)
                if other and other["id"] != it["id"]:
                    print(f"There's already an item called {other['name']}")
                    return 1
                print(f"Renamed item {it['name']} to {st.rename_item(it['id'], a.new_name)}")
            elif pl:
                other = st.find_place(a.new_name)
                if other and other["id"] != pl["id"]:
                    print(f"There's already a place called {other['name']}")
                    return 1
                first = a.new_name.lower().split()[0]
                mine = True if first in ("my", "our") else False if first in ("the", "a") else None
                print(f"Renamed place {pl['name']} to {st.rename_place(pl['id'], a.new_name, mine)}")
            else:
                print(f"Nothing called {a.name!r}")
                return 1
        else:
            if a.command in ("forget-people", "reset"):
                print(f"Forgot {st.delete_all_people()} people")
            if a.command in ("forget-items", "reset"):
                print(f"Forgot {st.delete_all_items()} items")
            if a.command in ("forget-places", "reset"):
                print(f"Forgot {st.delete_all_places()} places")
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
