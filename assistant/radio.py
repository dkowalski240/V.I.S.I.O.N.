"""Wi-Fi and Bluetooth context, read on a background thread.

Every `places.scan_interval_s` seconds:

* Wi-Fi (`places.wifi`): the network the laptop is connected to: SSID (the
  network name, e.g. "CampusNet") and BSSID (the specific access point, which
  in a big building narrows it down to a wing or floor). Windows: `netsh wlan
  show interfaces`. Since Windows 11 24H2 this needs Location services on
  (Settings > Privacy & security > Location, and "Let desktop apps access your
  location"); without it Windows hides the network name and this logs a hint once.
  Linux: nmcli.
* Bluetooth LE (`places.bluetooth`, off by default): a short scan with bleak.
  Fixed devices (TVs, speakers, printers, smart plugs, a desktop PC) make a
  fingerprint of the room. Phones, AirPods and AirTags change their address
  every few minutes for privacy, so they can't be tracked or identified this way.

Nothing here leaves the laptop. Only the network name and access-point id are
stored with a sighting or a saved place.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import PlacesConfig

log = logging.getLogger(__name__)


@dataclass
class RadioSnapshot:
    ts: float = 0.0
    ssid: str = ""
    bssid: str = ""
    signal: Optional[int] = None                         # percent
    ble: Dict[str, int] = field(default_factory=dict)    # address -> RSSI (dBm)


def parse_netsh(text: str) -> Dict[str, str]:
    """`netsh wlan show interfaces` -> {ssid, bssid, signal} (first connected interface)."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
        if not m:
            continue
        key, val = m.group(1).strip().lower(), m.group(2)
        if key == "ssid" and "ssid" not in out:
            out["ssid"] = val
        elif key in ("bssid", "ap bssid") and "bssid" not in out:
            out["bssid"] = val.lower()
        elif key == "signal" and "signal" not in out:
            out["signal"] = val.rstrip("%").strip()
    return out


def parse_nmcli(text: str) -> Dict[str, str]:
    """`nmcli -t -f ACTIVE,SSID,BSSID,SIGNAL dev wifi` -> the active network."""
    for line in text.splitlines():
        parts = re.split(r"(?<!\\):", line)
        if len(parts) >= 4 and parts[0] == "yes":
            return {"ssid": parts[1].replace("\\:", ":"), "bssid": parts[2].replace("\\:", ":").lower(),
                    "signal": parts[-1]}
    return {}


def read_wifi() -> Dict[str, str]:
    try:
        if sys.platform.startswith("win"):
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            r = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5,
                               creationflags=flags)
            text = r.stdout + r.stderr
            if "location" in text.lower() and "permission" in text.lower():
                return {"error": "location"}
            return parse_netsh(text)
        if sys.platform.startswith("linux"):
            r = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID,BSSID,SIGNAL", "dev", "wifi"], capture_output=True,
                               text=True, timeout=5)
            return parse_nmcli(r.stdout)
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    return {}


def scan_ble(seconds: float, min_rssi: int) -> Dict[str, int]:
    import asyncio

    from bleak import BleakScanner

    async def run():
        found = await BleakScanner.discover(timeout=seconds, return_adv=True)
        return {addr: int(adv.rssi) for addr, (_dev, adv) in found.items() if adv.rssi is not None}

    return {a: r for a, r in asyncio.run(run()).items() if r >= min_rssi}


class RadioScanner:
    def __init__(self, cfg: PlacesConfig):
        self.cfg = cfg
        self._snap = RadioSnapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wifi_ok = cfg.wifi
        self._ble_ok = cfg.bluetooth
        self._warned = False

    def latest(self) -> RadioSnapshot:
        with self._lock:
            return self._snap

    def scan_once(self) -> RadioSnapshot:
        snap = RadioSnapshot(ts=time.time())
        if self._wifi_ok:
            w = read_wifi()
            if w.get("error") == "location":
                if not self._warned:
                    log.warning("Windows hides the Wi-Fi name until Location is on: Settings > Privacy & security > "
                                "Location (and 'Let desktop apps access your location'). Wi-Fi context is off.")
                    self._warned = True
                self._wifi_ok = False
            elif "error" in w:
                log.info("Wi-Fi context unavailable: %s", w["error"])
                self._wifi_ok = False
            else:
                snap.ssid, snap.bssid = w.get("ssid", ""), w.get("bssid", "")
                snap.signal = int(w["signal"]) if w.get("signal", "").isdigit() else None
        if self._ble_ok:
            try:
                snap.ble = scan_ble(self.cfg.ble_scan_s, self.cfg.ble_min_rssi)
            except Exception as e:
                log.warning("Bluetooth scan unavailable (%s); Bluetooth context is off", e)
                self._ble_ok = False
        with self._lock:
            self._snap = snap
        return snap

    def start(self) -> "RadioScanner":
        if not (self._wifi_ok or self._ble_ok):
            return self
        self._thread = threading.Thread(target=self._run, name="radio", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = self.scan_once()
                log.debug("Radio: ssid=%r bssid=%s ble=%d", s.ssid, s.bssid, len(s.ble))
            except Exception:
                log.exception("Radio scan failed")
            self._stop.wait(self.cfg.scan_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.cfg.ble_scan_s + 6)


class StaticRadio:
    """Tests / no radio: returns a fixed snapshot."""

    def __init__(self, snap: Optional[RadioSnapshot] = None):
        self.snap = snap or RadioSnapshot()

    def latest(self) -> RadioSnapshot:
        return self.snap

    def start(self):
        return self

    def stop(self):
        pass
