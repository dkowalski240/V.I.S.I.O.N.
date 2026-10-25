"""Intent router: turns what was heard and seen into actions and speech.

Inputs (bus): speech_heard, person_seen.
Outputs (bus): intent (for the log), say.

Who may do what:
* push-to-talk / typed speech -> commands (parse_command)
* ambient speech               -> introductions and yes/no answers only, so
                                  random conversation can't trigger commands
While an enrollment waits for consent, an answer from either source counts.

Queries are answered by reading module state directly (who is in view, the
sighting log); actions go to the module that owns them.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .. import events as ev
from .. import geometry
from ..bus import Bus
from ..clock import Clock
from ..config import Config
from . import phrases
from .parser import Intent, clean_item, parse_answer, parse_command, parse_introduction

log = logging.getLogger(__name__)


class Router:
    def __init__(self, cfg: Config, bus: Bus, storage, clock: Optional[Clock] = None, faces=None,
                 objects=None, enrollment=None, output=None, vlm=None):
        self.cfg = cfg
        self.bus = bus
        self.storage = storage
        self.clock = clock or Clock()
        self.faces = faces
        self.objects = objects
        self.enrollment = enrollment
        self.output = output
        self.vlm = vlm
        self._sub = bus.subscribe(ev.SPEECH_HEARD, ev.PERSON_SEEN, maxsize=128)
        self._pending: Optional[dict] = None   # a yes/no confirmation we're waiting for
        self._last_item: Optional[str] = None  # for "describe it"
        self.last_heard: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ thread
    def start(self) -> "Router":
        self._thread = threading.Thread(target=self._run, name="router", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            msg = self._sub.get(timeout=0.1)
            try:
                if msg is not None:
                    self.handle(msg)
                self.tick()
            except Exception:
                log.exception("Router error")

    def process_pending(self) -> None:
        """Handle everything queued (tests call this instead of running the thread)."""
        for msg in self._sub.drain():
            self.handle(msg)
        self.tick()

    def tick(self) -> None:
        now = self.clock.now()
        if self.enrollment is not None:
            self.enrollment.tick(now)
        if self._pending and now > self._pending["deadline"]:
            self._say(f"Kept {self._pending['name']}.")
            self._pending = None

    # ------------------------------------------------------------------ helpers
    def _say(self, text: str, priority: int = ev.PRI_ANSWER, tone: Optional[str] = None, pan: float = 0.0,
             kind: str = ev.KIND_ANSWER, max_age_s: Optional[float] = None) -> None:
        fields = dict(text=text, priority=priority, pan=pan, kind=kind, tone=tone)
        if max_age_s is not None:
            fields["max_age_s"] = max_age_s
        self.bus.publish(ev.SAY, **fields)

    # ------------------------------------------------------------------ dispatch
    def handle(self, msg: dict) -> None:
        if msg["type"] == ev.PERSON_SEEN:
            self._on_person_seen(msg)
        elif msg["type"] == ev.SPEECH_HEARD:
            self._on_speech(msg)

    def _on_person_seen(self, m: dict) -> None:
        text = phrases.person_announcement(m["name"], m["clock_direction"], m["approaching"], m["sure"],
                                           self.cfg.direction.style)
        if m["approaching"]:
            self._say(text, ev.PRI_APPROACH, pan=m["pan"], kind=ev.KIND_APPROACH,
                      max_age_s=self.cfg.faces.announce.max_age_s)
        else:
            self._say(text, ev.PRI_PERSON, pan=m["pan"], kind=ev.KIND_PERSON,
                      max_age_s=self.cfg.faces.announce.max_age_s)

    def _on_speech(self, m: dict) -> None:
        text, source = (m.get("text") or "").strip(), m.get("source", "ambient")
        if not text:
            return
        self.last_heard = m
        now = self.clock.now()
        enr = self.enrollment

        # 1. An enrollment waiting for consent takes answers from anyone.
        if enr is not None and enr.awaiting_answer():
            ans = parse_answer(text)
            if ans is not None:
                self.bus.publish(ev.INTENT, type=f"consent_{ans.type}", args=ans.args, source=source)
                enr.on_answer(ans, now)
                return
            if source == "ambient":
                return

        if source == "ambient":
            # 2. Ambient speech: introductions only.
            name = parse_introduction(text)
            if name and enr is not None:
                self.bus.publish(ev.INTENT, type="introduction", args={"name": name}, source=source)
                enr.start_from_introduction(name, m.get("timestamp", now), m.get("t_end", now), source="ambient")
            return

        # 3. Push-to-talk / typed: commands.
        if source == "typed" and text.lower() in ("!reset-people", "!forget-everyone"):
            n = self.forget_all_people()  # testing shortcut: no confirmation
            self._say(f"Forgot {n} {'person' if n == 1 else 'people'}.", tone=ev.TONE_DONE)
            return
        intent = parse_command(text)
        self.bus.publish(ev.INTENT, type=intent.type, args=intent.args, source=source)
        log.info("Intent %s %s  <- %r", intent.type, intent.args, text)
        self.dispatch(intent, now)

    def dispatch(self, intent: Intent, now: float) -> None:
        t, a = intent.type, intent.args

        if t in ("yes", "no") and self._pending is not None:
            self._resolve_pending(t == "yes")
            return
        if t == "yes" or t == "no":
            self._say("Nothing to confirm.")
            return
        if t == "empty":
            return
        if t == "cancel":
            if self.enrollment is not None and self.enrollment.cancel():
                return
            if self._pending is not None:
                self._resolve_pending(False)
                return
            if self.objects is not None and self.objects.cancel_enrollment():
                self._say("Cancelled.")
                return
            self._say("Okay.")
            return

        if t == "done":
            if self.objects is not None and self.objects.finish_enrollment_early(now):
                return  # the capture ends on the next frame and says "Saved ..."
            self._say("Okay.")
            return

        handler = getattr(self, f"_do_{t}", None)
        if handler is None:
            self._say(phrases.DIDNT_CATCH, tone=ev.TONE_ERROR)
            return
        handler(a, now)

    # ------------------------------------------------------------------ people
    def _do_who_here(self, a, now):
        if self.faces is None:
            self._say("Face recognition is off.")
            return
        tracks = self.faces.who_is_here(now)
        self._say(phrases.who_is_here(tracks, self.cfg.faces.matching.t_sure, self.cfg.direction.style))

    def _do_save_person(self, a, now):
        if self.enrollment is None:
            self._say("Face recognition is off.")
            return
        if self.enrollment.busy():
            self._say("Still saving someone.")
            return
        self.enrollment.start_manual(a["name"], now)

    def _do_list_people(self, a, now):
        self._say(phrases.list_names([p["name"] for p in self.storage.list_people()], "people"))

    def _do_correction(self, a, now):
        if self.enrollment is not None and self.enrollment.handle_correction(a["name"], now):
            return
        self._say("Nothing to correct.")

    _HERE = ("this room", "this place", "here", "this spot", "the room i'm in", "the room i am in")

    def _do_rename(self, a, now):
        """"rename Sierra to Sarah" / "rename my wallet to my black wallet" /
        "rename the study to my office" / "rename this room to the guest room"."""
        old_raw = (a.get("old_raw") or a.get("old") or "").lower()
        new_raw = (a.get("new_raw") or a.get("new") or "").lower()
        person = self.storage.find_person(a["old"]) if a.get("old") else None
        if person is not None:
            if not a.get("new"):
                self._say("That doesn't sound like a name.")
                return
            self.storage.rename_person(person["id"], a["new"])
            if self.faces is not None:
                self.faces.reload_gallery()
            self._say(f"Renamed to {a['new']}.", tone=ev.TONE_DONE)
            return
        old, new = clean_item(old_raw), clean_item(new_raw)
        if not new:
            self._say("Rename it to what?")
            return
        if old_raw in self._HERE or old in self._HERE:
            guess = self.objects.current_place() if self.objects is not None else None
            if guess is None:
                self._say("I don't recognise this place.")
                return
            self._rename_place(self.storage.find_place(guess), new, new_raw)
            return
        item = self.storage.find_item(old)
        if item is not None:
            other = self.storage.find_item(new)
            if other is not None and other["id"] != item["id"]:
                self._say(f"You already have an item called {other['name']}. Forget one of them first.")
                return
            stored = self.storage.rename_item(item["id"], new)
            if self.objects is not None:
                self.objects.reload_gallery()
            if self._last_item == item["name"]:
                self._last_item = stored
            self._say(f"Renamed your {item['name']} to {stored}.", tone=ev.TONE_DONE)
            return
        place = self.storage.find_place(old)
        if place is not None:
            self._rename_place(place, new, new_raw)
            return
        self._say(f"I don't know {old or old_raw}.")

    def _rename_place(self, place, new, new_raw):
        other = self.storage.find_place(new)
        if other is not None and other["id"] != place["id"]:
            self._say(f"You already have {self.storage.place_label(other['name'])}. Forget one of them first.")
            return
        det = new_raw.split()[0] if new_raw.split() else ""
        mine = True if det in ("my", "our") else False if det in ("the", "a") else None
        before = self.storage.place_label(place["name"])
        stored = self.storage.rename_place(place["id"], new, mine)
        if self.objects is not None:
            self.objects.reload_places()
        self._say(f"Renamed {before} to {self.storage.place_label(stored)}.", tone=ev.TONE_DONE)

    # ------------------------------------------------------------------ forget (with confirmation)
    def _do_forget(self, a, now):
        target = a["target"]
        person = self.storage.find_person(target)
        item = None if person else self.storage.find_item(target)
        place = None if (person or item) else self.storage.find_place(target)
        if person is not None:
            pending = {"action": "person", "id": person["id"], "name": person["name"]}
        elif item is not None:
            pending = {"action": "item", "id": item["id"], "name": f"your {item['name']}"}
        elif place is not None:
            pending = {"action": "place", "id": place["id"], "name": self.storage.place_label(place["name"])}
        else:
            self._say(f"I don't know {target}.")
            return
        self._confirm(pending, now)

    def _do_forget_all(self, a, now):
        scope = a["scope"]
        counts = {"people": len(self.storage.list_people()), "items": len(self.storage.list_items()),
                  "places": len(self.storage.list_places())}
        if scope == "everything":
            if not any(counts.values()):
                self._say("Nothing saved.")
                return
            name = "everything"
        else:
            n = counts[scope]
            if n == 0:
                self._say({"people": "I don't know anyone yet.", "items": "No items saved yet.",
                           "places": "No places saved yet."}[scope])
                return
            unit = {"people": ("person", "people"), "items": ("item", "items"), "places": ("place", "places")}[scope]
            name = f"all {n} {unit[0] if n == 1 else unit[1]}"
        self._confirm({"action": "all", "scope": scope, "name": name}, now)

    def _confirm(self, pending: dict, now: float) -> None:
        if not self.cfg.commands.confirm_forget:
            self._pending = pending
            self._resolve_pending(True)
            return
        pending["deadline"] = now + self.cfg.commands.confirm_timeout_s
        self._pending = pending
        self._say(f"Forget {pending['name']}? Say yes.")

    def forget_all_people(self) -> int:
        """Delete every enrolled person right away (console !reset-people, --forget-people)."""
        n = self.storage.delete_all_people()
        if self.faces is not None:
            self.faces.reload_gallery()
            self.faces.forget_all_cooldowns()
        self.bus.publish(ev.PERSON_FORGOTTEN, person_id=None, name="*", count=n)
        return n

    def _resolve_pending(self, yes: bool) -> None:
        p, self._pending = self._pending, None
        if p is None:
            return
        if not yes:
            self._say(f"Kept {p['name']}.")
            return
        action = p["action"]
        if action == "person":
            self.storage.delete_person(p["id"])
            if self.faces is not None:
                self.faces.reload_gallery()
                self.faces.forget_cooldowns(p["id"])
            self.bus.publish(ev.PERSON_FORGOTTEN, person_id=p["id"], name=p["name"])
        elif action == "item":
            self.storage.delete_item(p["id"])
            if self.objects is not None:
                self.objects.reload_gallery()
            self.bus.publish(ev.ITEM_FORGOTTEN, item_id=p["id"], name=p["name"])
        elif action == "place":
            self.storage.delete_place(p["id"])
            if self.objects is not None:
                self.objects.reload_places()
        elif action == "all":
            scope = p["scope"]
            if scope in ("people", "everything"):
                self.forget_all_people()
            if scope in ("items", "everything"):
                self.storage.delete_all_items()
                if self.objects is not None:
                    self.objects.reload_gallery()
                self.bus.publish(ev.ITEM_FORGOTTEN, item_id=None, name="*")
            if scope in ("places", "everything"):
                self.storage.delete_all_places()
                if self.objects is not None:
                    self.objects.reload_places()
        self._say(f"Forgot {p['name']}.", tone=ev.TONE_DONE)

    # ------------------------------------------------------------------ places
    def _do_set_place(self, a, now):
        if self.objects is None:
            self._say("Object memory is off.")
            return
        self.objects.start_place_enrollment(a["place"], now, mine=a.get("mine"))

    def _do_direction_style(self, a, now):
        style = a["style"]
        self.cfg.direction.style = style  # shared config object: every module says it this way from now on
        if style == "clock":
            self._say("Clock directions. 12 o'clock is straight ahead, 3 is to your right, 9 is to your left.")
        elif style == "both":
            self._say("Clock directions with left and right. For example: 2 o'clock, to your right.")
        else:
            self._say("Okay, I'll say left and right.")

    def _do_where_am_i(self, a, now):
        guess = self.objects.current_place() if self.objects is not None else None
        if guess is None:
            n = len(self.storage.list_places())
            self._say("I don't recognise this place." if n else "No places saved yet. Say: this is the kitchen.")
        else:
            self._say(f"Probably {self.storage.place_label(guess)}.")

    def _do_list_places(self, a, now):
        names = [p["name"] for p in self.storage.list_places()]
        if not names:
            self._say("No places saved yet.")
        else:
            self._say("Saved places: " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])
                      + ".")

    # ------------------------------------------------------------------ items
    def _do_remember_item(self, a, now):
        if self.objects is None:
            self._say("Object memory is off.")
            return
        self.objects.start_enrollment(a["item"], now)

    def _do_list_items(self, a, now):
        self._say(phrases.list_names([i["name"] for i in self.storage.list_items()], "items"))

    def _do_where_item(self, a, now):
        name = a["item"]
        item = self.storage.find_item(name)
        if item is None:
            person = self.storage.find_person(name)
            if person is not None:
                self._where_person(person, now)
                return
            self._say(f"I don't know your {name} yet.")
            return
        self._last_item = item["name"]
        hour = self.objects.in_view(item["id"], now) if self.objects is not None else None
        s = self.storage.last_sighting(item["id"])
        network = ""
        if s is not None and s.wifi_ssid and self.cfg.places.mention_network and self.objects is not None:
            current = self.objects.radio.latest().ssid
            if current and current != s.wifi_ssid:  # it was left somewhere else
                network = s.wifi_ssid
        self._say(phrases.item_location(item["name"], None if s is None else now - s.ts, s.context if s else "",
                                        hour, place=self.storage.place_label(s.place) if s else "",
                                        companions=s.companions if s else "", network=network,
                                        near=s.near if s else "", style=self.cfg.direction.style))
        if a.get("detail") and s is not None:
            self._describe(item["name"], s)

    def _where_person(self, person, now):
        if self.faces is not None:
            for t in self.faces.who_is_here(now):
                if t.get("person_id") == person["id"]:
                    self._say(f"{person['name']} {geometry.at_phrase(t['hour'], self.cfg.direction.style)}.")
                    return
        self._say(f"I don't see {person['name']} right now.")

    def _do_describe(self, a, now):
        name = a.get("item") or self._last_item
        if not name:
            self._say("Describe what?")
            return
        item = self.storage.find_item(name)
        s = self.storage.last_sighting(item["id"]) if item else None
        if s is None:
            self._say(f"I haven't seen your {name} yet.")
            return
        self._describe(item["name"], s)

    def _describe(self, item_name: str, sighting) -> None:
        if self.vlm is None or not self.cfg.cloud.vlm_enabled:
            self._say("Detailed descriptions are off.")
            return

        def work():
            text = self.vlm.describe(item_name, sighting)
            self._say(text or "Couldn't get a description.")

        threading.Thread(target=work, name="vlm", daemon=True).start()

    # ------------------------------------------------------------------ speech controls
    def _do_repeat(self, a, now):
        if self.output is not None:
            self.output.repeat()

    def _do_quiet_on(self, a, now):
        if self.output is not None:
            self.output.set_quiet(True)
        self._say("Quiet mode.")

    def _do_quiet_off(self, a, now):
        if self.output is not None:
            self.output.set_quiet(False)
        self._say("Quiet mode off.")

    def _do_help(self, a, now):
        self._say(phrases.HELP)

    def _do_unknown(self, a, now):
        self._say(phrases.DIDNT_CATCH, tone=ev.TONE_ERROR)
