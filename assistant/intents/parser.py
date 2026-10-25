"""Rule-based parsing of what was heard. Pure functions, unit-tested.

Three parsers, used in different situations:

* parse_command(text): push-to-talk or typed input. "who's here?",
  "where's my wallet?", "remember this, it's my keys", "save this person as
  Sarah", "forget Sarah", "list people", "repeat that", "quiet mode"...
* parse_introduction(text): ambient speech. "Hi, I'm Sarah", "My name is
  Sarah", "I'm Sarah, nice to meet you", "This is my friend Sarah". Returns a
  name or None. "I'm fine" or "I'm going to the store" must NOT count, so
  weak patterns ("I'm X") need a capitalised name (Whisper capitalises proper
  nouns) or a greeting nearby, and X must not be a common word.
* parse_answer(text): ambient or push-to-talk while waiting for consent.
  "Yeah, sure" -> yes, "No thanks" -> no, "No problem" -> yes,
  "It's Sara, actually" -> name correction. Questions ("Is it okay if...?")
  are ignored, so the user asking the question aloud doesn't count as a yes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------- words
_NAME_STOP = set("""
a an the so just not very really here there fine good great okay ok alright sorry sure glad happy
tired done back home ready also still in on at from with about like kind afraid excited new late
early busy hungry thirsty sick cold hot free well all too more your his her their our my me you
him them us it that this what who which where when why how um uh er yes no yeah nope cool nice
awesome lost confused serious kidding joking positive certain aware familiar used supposed allowed
able right left almost nearly already finally honored impressed thrilled worried scared sad mad
angry bored single married older younger fun funny curious interested pretty actually literally
totally over out up down off into for to and or but if then than because gonna wanna kinda sorta
one two three four five six seven eight nine ten hundred staff student here's there's going doing
trying looking coming leaving talking saying thinking working getting being having making taking
waiting staying calling asking telling studying sitting standing walking running heading off
afraid good-looking okay-ish fine-ish the-best best worst fan big small tall short hungry starving
cool fine meet nice pleased delighted happy thanks thank please hi hello hey bye goodbye welcome
always never sometimes usually probably maybe definitely absolutely obviously basically honestly
someone somebody anyone anybody everyone everybody nobody nothing something everything
""".split())

_GREETING = re.compile(
    r"\b(hi|hello|hey|hiya|howdy|nice to meet you|pleased to meet you|good to meet you|pleasure|"
    r"good (morning|afternoon|evening)|what'?s up)\b",
    re.I,
)

_RELATION = (r"(?:friend|colleague|coworker|co-worker|teammate|roommate|partner|classmate|neighbor|"
             r"neighbour|sister|brother|mom|mum|dad|mother|father|wife|husband|girlfriend|boyfriend|"
             r"cousin|aunt|uncle|son|daughter|boss|manager|professor|teacher|buddy|pal)")

_YES = re.compile(
    r"^(yes|yeah|yea|yep|yup|ya|sure|okay|ok|alright|all right|absolutely|definitely|certainly|"
    r"of course|go ahead|go for it|do it|sounds good|that'?s fine|that'?s okay|fine|please|"
    r"why not|uh huh|mm hmm|totally|no problem|no worries|i don'?t mind|not a problem|correct|"
    r"that'?s right|you can|you may|i agree|i consent)\b"
)
_NO = re.compile(
    r"^(no|nope|nah|don'?t|do not|please don'?t|rather not|i'?d rather not|not really|no thanks|"
    r"no thank you|i don'?t think so|never mind|nevermind|cancel|stop|i don'?t want|i do not want)\b"
)
_QUESTION_START = {"is", "are", "can", "could", "may", "do", "does", "would", "will", "should",
                   "shall", "did", "what", "who", "where", "why", "how", "mind", "okay if", "ok if"}
# The user asking the question aloud ("alright if I save you", "okay with you if...")
_ASKING = re.compile(
    r"^(?:okay|ok|alright|all right|fine|cool|good|sure)[\s,]+(?:if|with you|to (?:save|remember|keep))\b"
    r"|\b(?:remember you|save you|remember your face|save your face|keep your face)\b"
)

_LEADING_FILLER = re.compile(
    r"^(?:(?:hey|okay|ok|um|uh|so|please|assistant|hey assistant|can you|could you|would you|"
    r"tell me|and|well|alright|now)[\s,]+)+"
)
_TRAILING_FILLER = re.compile(r"(?:[\s,]+(?:please|now|again|for me|thanks|thank you|right now|today))+$")


# Room words: "this is the kitchen" / "this is my bedroom" name a place, not an item.
_PLACE_WORDS = (
    r"(?:(?:\w+'s |\w+ )?(?:kitchen|kitchenette|bedroom|bathroom|living room|lounge|dining room|office|study|garage|"
    r"hallway|hall|basement|attic|porch|balcony|garden|yard|backyard|lab|classroom|library|car|dorm|dorm room|"
    r"laundry room|entryway|foyer|closet|workshop|studio|cafeteria|gym|lobby|conference room|meeting room|"
    r"break room|room|apartment|house|home|desk))"
)


@dataclass
class Intent:
    type: str
    args: dict = field(default_factory=dict)
    text: str = ""


# ----------------------------------------------------------------------------- helpers
def normalize(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").lower()
    t = re.sub(r"[^a-z0-9' ,]+", " ", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    return t


def _strip_fillers(t: str) -> str:
    t = _LEADING_FILLER.sub("", t)
    t = _TRAILING_FILLER.sub("", t)
    return t.strip(" ,")


def clean_name(raw: str, require_capital: bool = False) -> Optional[str]:
    """First word (plus a second capitalised word, "Mary Jane") of `raw`, title-cased.
    Returns None if it doesn't look like a name."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", raw.replace("’", "'"))
    if not words:
        return None
    first = words[0]
    low = first.lower().strip("'-")
    if low in _NAME_STOP or len(low) < 2:
        return None
    if low.endswith("ing") and len(low) > 5:
        return None
    if require_capital and not first[0].isupper():
        return None
    parts = [first]
    if len(words) > 1:
        second = words[1]
        if second[0].isupper() and second.lower() not in _NAME_STOP and len(second) > 1:
            parts.append(second)
    return " ".join(_title(p) for p in parts)


def _title(word: str) -> str:
    """sarah/SARAH -> Sarah; keep mixed case like McKenzie."""
    if word.islower() or word.isupper():
        return word.capitalize()
    return word[0].upper() + word[1:]


def clean_item(raw: str) -> str:
    """'my wallet' -> 'wallet', 'the guest room' -> 'guest room'."""
    return _clean_item(raw)


def _clean_item(raw: str) -> str:
    t = raw.strip(" ,?.!")
    t = re.sub(r"^(?:my|the|a|an|our|these|those|this|that)\s+", "", t)
    t = re.sub(r"\s+(?:is|are|at|go|gone|went|now|please|again)$", "", t)
    t = re.sub(r"\s+(?:is|are)$", "", t)
    return t.strip(" ,")


# ----------------------------------------------------------------------------- introductions
_INTRO_STRONG = [
    re.compile(r"\bmy name(?:'s| is)\s+(?P<name>[^,.!?;]+)", re.I),
    re.compile(r"\b(?:you can )?call me\s+(?P<name>[^,.!?;]+)", re.I),
    re.compile(r"\bthe name(?:'s| is)\s+(?P<name>[^,.!?;]+)", re.I),
    re.compile(r"\bthis is my " + _RELATION + r",?\s+(?P<name>[^,.!?;]+)", re.I),
    re.compile(r"\bmeet my " + _RELATION + r",?\s+(?P<name>[^,.!?;]+)", re.I),
]
_INTRO_WEAK = [
    re.compile(r"\b(?:i'm|i am|im)\s+(?P<name>[^,.!?;]+)", re.I),
    re.compile(r"\bthis is\s+(?P<name>[^,.!?;]+)", re.I),
    re.compile(r"\b(?:it's|its)\s+(?P<name>[^,.!?;]+)\s+(?:here|by the way)\b", re.I),
]


def parse_introduction(text: str) -> Optional[str]:
    text = text.replace("’", "'").strip()
    if not text:
        return None
    for pat in _INTRO_STRONG:
        m = pat.search(text)
        if m:
            name = clean_name(m.group("name"))
            if name:
                return name
    greeting = bool(_GREETING.search(text))
    for pat in _INTRO_WEAK:
        m = pat.search(text)
        if not m:
            continue
        raw = m.group("name").strip()
        first = raw.split()[0] if raw.split() else ""
        if first.lower() in ("a", "an", "the", "my", "your", "not", "so", "just"):
            continue
        name = clean_name(raw, require_capital=not greeting)
        if name:
            return name
    return None


# ----------------------------------------------------------------------------- answers
_CORRECTION = re.compile(
    r"^(?:(?:no|nope|actually|sorry|wait|oh)[\s,]+)*(?:it'?s|it is|my name is|my name'?s|i'?m|i am|call me|"
    r"her name is|his name is|their name is|the name is|the name'?s|name'?s|no it'?s)\s+(?P<name>.+)$",
    re.I,
)


def parse_answer(text: str) -> Optional[Intent]:
    """yes / no / name correction, or None if it isn't an answer."""
    raw = text.replace("’", "'").strip()
    t = normalize(raw)
    if not t:
        return None
    words = t.replace(",", "").split()
    if raw.rstrip().endswith("?") or words[0] in _QUESTION_START or " ".join(words[:2]) in _QUESTION_START:
        return None
    if _ASKING.search(t):  # "alright if it remembers you" without a question mark
        return None
    m = _CORRECTION.match(raw.strip(" .!"))
    if m:
        name = clean_name(m.group("name"))
        if name:
            return Intent("correction", {"name": name}, text)
    if len(words) > 8:
        return None
    if _YES.match(t):
        return Intent("yes", {}, text)
    if _NO.match(t):
        return Intent("no", {}, text)
    return None


# ----------------------------------------------------------------------------- commands
_PATTERNS = [
    ("cancel", re.compile(r"^(?:cancel|never ?mind|forget it|stop)$")),
    ("done", re.compile(r"^(?:done|i'?m done|all done|finished|i'?m finished|that'?s it|that'?s all|"
                        r"that'?s the whole room|ok(?:ay)? done)$")),
    ("save_person", re.compile(
        r"^(?:save|remember|add|enroll|learn|store)\s+(?:this|that)\s+(?:person|guy|girl|woman|man|face|friend|one)"
        r"\s+(?:as|is|called)\s+(?P<name>.+)$")),
    ("save_person", re.compile(r"^(?:save|remember|add|enroll|learn)\s+(?:him|her|them)\s+as\s+(?P<name>.+)$")),
    ("save_person", re.compile(r"^(?:this person is|this person's name is|their name is)\s+(?P<name>.+)$")),
    ("rename", re.compile(r"^(?:rename|change)\s+(?:the\s+)?name\s+of\s+(?P<old>.+?)\s+to\s+(?P<new>.+)$")),
    ("rename", re.compile(r"^(?:rename|change)\s+(?P<old>.+?)\s+(?:to|as|into)\s+(?P<new>.+)$")),
    ("correction", re.compile(
        r"^(?:no,?\s+)?(?:her name is|his name is|their name is|the name is|actually,?\s+it'?s|i said|"
        r"no,?\s+it'?s)\s+(?P<name>.+)$")),
    ("forget_all", re.compile(
        r"^(?:forget|delete|erase|remove)\s+(?P<scope>everyone|everybody|all (?:the |my )?(?:people|faces|names|persons)"
        r"|all (?:the |my )?(?:items|things|objects|stuff)|all (?:the |my )?(?:places|rooms)|everything)$")),
    ("forget", re.compile(r"^(?:forget|delete|remove|erase|unlearn)\s+(?:about\s+)?(?P<target>.+)$")),
    ("where_am_i", re.compile(r"^(?:where am i|where are we|what room is this|which room is this|what place is this|"
                              r"what room am i in)$")),
    ("list_places", re.compile(r"^(?:list (?:the |my |all )?(?:places|rooms)|what (?:places|rooms) do you know)$")),
    ("set_place", re.compile(
        r"^(?:remember|save|learn|call)\s+this\s+(?:place|room|spot|area)\s+(?:as\s+)?(?P<det>the |my |our |a )?"
        r"(?P<place>.+)$")),
    ("set_place", re.compile(
        r"^(?:this is|this room is|we'?re in|we are in|i'?m in|i am in)\s+(?P<det>the |my |our |a )?(?P<place>"
        + _PLACE_WORDS + r")$")),
    ("direction_style", re.compile(
        r"^(?:use|switch to|say|give me|i want)\s+(?:the\s+)?(?P<style>clock|clock face|clock positions?|clock directions?|"
        r"clock and words|both|both directions|left and right|left right|left or right|simple directions|plain directions|"
        r"words|words for directions|normal directions)(?:\s+directions)?(?:\s+please)?$")),
    ("list_people", re.compile(
        r"^(?:list(?: the| all)? people|who do you know|which people do you know|who have you met|"
        r"list (?:faces|names)|people list|who do you remember)$")),
    ("list_items", re.compile(
        r"^(?:list(?: my| the| all)? (?:items|things|objects|stuff)|what (?:items|things|objects|stuff) do you "
        r"(?:know|remember)|what do you remember)$")),
    ("remember_item", re.compile(
        r"^(?:remember|save|learn|memorize|memorise)\s+(?:this|that|these|it)\s*(?:as\s+|,\s*|is\s+)?"
        r"(?:it'?s\s+|it is\s+|this is\s+|these are\s+|they'?re\s+|they are\s+)?(?P<item>.+)$")),
    ("remember_item", re.compile(r"^(?:remember|learn|memorize)\s+my\s+(?P<item>.+)$")),
    ("remember_item", re.compile(r"^(?:this is|these are)\s+my\s+(?P<item>.+)$")),
    ("describe", re.compile(r"^(?:describe it|describe that|more detail|more details|tell me more)$")),
    ("describe", re.compile(
        r"^(?:describe|tell me more about|more details? (?:on|about))\s+(?:where\s+)?(?P<item>.+?)(?:\s+is|\s+are)?$")),
    ("where_item", re.compile(r"^where did (?P<item>.+?) go$")),
    ("where_item", re.compile(
        r"^(?:where(?:'s| is| are|'re| were| was)|where did i (?:put|leave)|have you seen|did you see|find|locate)"
        r"\s+(?P<item>.+?)(?P<detail>,?\s+in (?:more )?detail)?$")),
    ("who_here", re.compile(
        r"^(?:who'?s|who is|who are|who'?re)\s+(?:here|around|there|nearby|near me|in front(?: of me)?|with me|"
        r"in the room|that|this|coming)\b")),
    ("who_here", re.compile(r"^(?:who do you see|who can you see|is anyone (?:here|there|around)|is anybody "
                            r"(?:here|there|around)|anyone (?:here|around)|who'?s here)$")),
    ("repeat", re.compile(
        r"^(?:repeat(?: that| it| yourself)?|say (?:that|it) again|again|what did you say|come again|pardon|"
        r"sorry what|what was that|what)$")),
    ("quiet_off", re.compile(
        r"^(?:quiet mode off|turn off quiet mode|turn quiet mode off|stop quiet mode|exit quiet mode|"
        r"you can talk(?: again)?|talk again|unmute|resume|normal mode|speak again)$")),
    ("quiet_on", re.compile(
        r"^(?:quiet mode(?: on)?|turn on quiet mode|turn quiet mode on|be quiet|quiet|silence|shh+|hush|mute|"
        r"stop talking)$")),
    ("help", re.compile(r"^(?:help|what can you do|commands|what can i say|options)$")),
]


def parse_command(text: str) -> Intent:
    t = _strip_fillers(normalize(text))
    if not t:
        return Intent("empty", {}, text)
    words = t.replace(",", "").split()
    if _PATTERNS[0][1].match(t.replace(",", "")):
        return Intent("cancel", {}, text)
    if len(words) <= 4:
        ans = parse_answer(text)
        if ans is not None and ans.type in ("yes", "no"):
            return Intent(ans.type, {}, text)
    t_plain = t.replace(",", "")
    for kind, pat in _PATTERNS:
        m = pat.match(t) or pat.match(t_plain)
        if not m:
            continue
        g = m.groupdict()
        if kind in ("save_person", "correction"):
            # Recover capitalisation from the original text when possible.
            name = clean_name(g["name"])
            if not name:
                continue
            return Intent(kind, {"name": name}, text)
        if kind == "rename":
            # A person, an item or a place: the router decides which. old/new are the
            # name-style versions ("Sarah"), old_raw/new_raw the words as said ("my wallet").
            old_raw, new_raw = g["old"].strip(" ,.!?"), g["new"].strip(" ,.!?")
            if not (_clean_item(old_raw) and _clean_item(new_raw)):
                continue
            return Intent(kind, {"old": clean_name(old_raw), "new": clean_name(new_raw),
                                 "old_raw": old_raw, "new_raw": new_raw}, text)
        if kind == "forget_all":
            sc = g["scope"]
            scope = ("people" if sc in ("everyone", "everybody") or any(w in sc for w in ("people", "faces", "names",
                                                                                          "persons"))
                     else "items" if any(w in sc for w in ("items", "things", "objects", "stuff"))
                     else "places" if any(w in sc for w in ("places", "rooms")) else "everything")
            return Intent(kind, {"scope": scope}, text)
        if kind == "set_place":
            place = _clean_item(g["place"])
            if not place:
                continue
            det = (g.get("det") or "").strip()
            # "this is my bedroom" -> later "your bedroom"; "this is the library" -> "the library"
            return Intent(kind, {"place": place, "mine": det in ("my", "our") if det else None}, text)
        if kind == "direction_style":
            st = g["style"]
            style = "both" if "both" in st or "and words" in st else "clock" if st.startswith("clock") else "words"
            return Intent(kind, {"style": style}, text)
        if kind == "forget":
            target = _clean_item(g["target"])
            if not target:
                continue
            return Intent(kind, {"target": target}, text)
        if kind == "remember_item":
            item = _clean_item(g["item"])
            if not item or item in ("person", "face", "guy", "girl"):
                continue
            return Intent(kind, {"item": item}, text)
        if kind == "where_item":
            item = _clean_item(g["item"])
            if not item:
                continue
            return Intent(kind, {"item": item, "detail": bool(g.get("detail"))}, text)
        if kind == "describe":
            item = _clean_item(g.get("item") or "")
            return Intent(kind, {"item": item or None}, text)
        return Intent(kind, {}, text)
    return Intent("unknown", {}, text)
