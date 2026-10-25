"""Event names and speech priorities shared by all modules.

Every message on the bus is a plain dict with at least {"type", "ts"}.
The main ones (fields as in the project brief):

    person_seen     name, person_id, track_id, clock_direction, angle_deg, pan,
                    approaching, confidence, sure, camera_id
    object_sighting item, item_id, timestamp, bbox, snapshot_path, context,
                    clock_direction, score, camera_id
    speech_heard    text, timestamp, t_end, source ("ambient" | "ptt" | "typed")
    intent          type, args, source
    say             text, priority, pan, kind, tone, max_age_s

Bookkeeping events: ptt (pressed/released), person_saved, person_forgotten,
item_saved, item_forgotten, status.
"""

PERSON_SEEN = "person_seen"
OBJECT_SIGHTING = "object_sighting"
SPEECH_HEARD = "speech_heard"
INTENT = "intent"
SAY = "say"

PTT = "ptt"
PERSON_SAVED = "person_saved"
PERSON_FORGOTTEN = "person_forgotten"
ITEM_SAVED = "item_saved"
ITEM_FORGOTTEN = "item_forgotten"
STATUS = "status"

# Speech priorities: lower number = spoken first.
PRI_APPROACH = 0   # a known person walking toward the user
PRI_ANSWER = 1     # answers to the user's questions, enrollment prompts
PRI_PERSON = 2     # a known person in view, not approaching
PRI_INFO = 3       # everything else

# Speech kinds decide what quiet mode keeps and how long a message stays relevant.
KIND_APPROACH = "approach"
KIND_PERSON = "person"
KIND_ANSWER = "answer"
KIND_PROMPT = "prompt"   # enrollment questions: never dropped, never interrupted
KIND_INFO = "info"

# Short tones used instead of words where words aren't needed.
TONE_LISTEN = "listen"     # push-to-talk pressed
TONE_DONE = "done"         # something saved
TONE_ERROR = "error"       # didn't understand / failed
TONE_PERSON = "person"     # quiet mode: a known person is approaching
TONE_CAPTURE = "capture"   # started capturing views
