"""Everything the assistant says, in one place. Short: the user keeps one ear
free, so announcements aim for about six words."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..geometry import at_phrase, clock_phrase, time_ago_phrase


def person_announcement(name: str, hour: int, approaching: bool, sure: bool, style: str = "clock") -> str:
    where = clock_phrase(hour, style)
    if approaching:
        return f"{name}, approaching, {where}" if sure else f"Maybe {name}, approaching, {where}"
    return f"That's {name}, {where}" if sure else f"I think that's {name}, {where}"


def who_is_here(tracks: List[Dict], t_sure: float, style: str = "clock") -> str:
    known = [t for t in tracks if t.get("name")]
    unknown = [t for t in tracks if not t.get("name")]
    parts = []
    for t in known:
        who = t["name"] if t.get("confidence", 0) >= t_sure else f"probably {t['name']}"
        s = f"{who} {at_phrase(t['hour'], style)}"
        if t.get("approaching"):
            s += ", approaching"
        parts.append(s)
    if unknown:
        n = len(unknown)
        parts.append("one person I don't know" if n == 1 else f"{n} people I don't know")
    if not parts:
        return "Nobody in view."
    return " ".join(p[0].upper() + p[1:] + "." for p in parts)


def list_names(names: List[str], what: str = "people") -> str:
    if not names:
        return "I don't know anyone yet." if what == "people" else "No items saved yet."
    if len(names) == 1:
        joined = names[0]
    else:
        joined = ", ".join(names[:-1]) + " and " + names[-1]
    return f"I know {joined}." if what == "people" else f"Saved items: {joined}."


def location_phrase(context: str, place: str = "", near: str = "") -> str:
    """Combine the item's own frame ("on the floor", "next to the laptop"), the room
    ("your bedroom") and a landmark from the frames around it ("bed"):
    "on the floor of your bedroom, near the bed"."""
    parts = [x.strip() for x in context.split(",") if x.strip()] if context else []
    on = parts[0] if parts and parts[0].startswith("on the ") else ""
    rest = parts[1:] if on else parts
    if near and (any(near in x for x in parts) or any(x.startswith("next to") for x in rest)):
        near = ""  # don't repeat it, and "next to the laptop" (same frame) already says enough
    if on and place:
        head = f"{on} {'of' if on in ('on the floor', 'on the ground') else 'in'} {place}"
    elif place:
        head = f"in {place}"
    else:
        head = on
    return ", ".join(x for x in [head, *rest, f"near the {near}" if near else ""] if x)


def item_location(item: str, seconds_ago: Optional[float], context: str, in_view_hour: Optional[int],
                  place: str = "", companions: str = "", network: str = "", near: str = "",
                  style: str = "clock") -> str:
    """e.g. "Your wallet was last seen 14 minutes ago, on the floor of your bedroom, near the bed.
    You were with Sarah." `place` is already worded ("your bedroom", "the library")."""
    are = item.endswith("s")
    if in_view_hour is not None:
        where = f"Your {item} {'are' if are else 'is'} in view {at_phrase(in_view_hour, style)}"
        return f"{where}, {context}." if context else f"{where}."
    if seconds_ago is None:
        return f"I haven't seen your {item} yet."
    text = f"Your {item} {'were' if are else 'was'} last seen {time_ago_phrase(seconds_ago)}"
    loc = location_phrase(context, place, near)
    text += f", {loc}." if loc else "."
    if companions:
        names = [n.strip() for n in companions.split(",") if n.strip()]
        text += " You were with " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]) + "."
    if network:
        text += f" That was on the {network} network."
    return text


HELP = "Ask who's here, or where's my item."
DIDNT_CATCH = "Sorry, didn't catch that."
