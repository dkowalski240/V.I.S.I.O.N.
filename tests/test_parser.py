import pytest

from assistant.intents.parser import parse_answer, parse_command, parse_introduction


@pytest.mark.parametrize("text,name", [
    ("Hi, I'm Sarah.", "Sarah"),
    ("My name is Sarah", "Sarah"),
    ("I'm Sarah, nice to meet you", "Sarah"),
    ("Hey! I am Marcus. Nice to meet you.", "Marcus"),
    ("hello, i'm priya", "Priya"),            # lower case but greeted
    ("This is my friend Sarah.", "Sarah"),
    ("You can call me Alex.", "Alex"),
    ("I'm Mary Jane, hi.", "Mary Jane"),
    ("Hi I'm Lisa and this is Tom", "Lisa"),
])
def test_introductions(text, name):
    assert parse_introduction(text) == name


@pytest.mark.parametrize("text", [
    "I'm fine, thanks.", "I'm going to the store.", "I'm so tired today", "I'm not sure",
    "I am good", "This is great!", "I'm sorry about that", "i'm here", "I'm looking for my keys",
    "This is my wallet", "It's okay", "Is it okay if I remember you?",
])
def test_not_introductions(text):
    assert parse_introduction(text) is None


@pytest.mark.parametrize("text,kind", [
    ("Sure.", "yes"), ("Yeah, go ahead.", "yes"), ("Yes please", "yes"), ("No problem!", "yes"),
    ("Of course", "yes"), ("No.", "no"), ("No thanks", "no"), ("I'd rather not.", "no"),
    ("Nope", "no"),
])
def test_answers(text, kind):
    assert parse_answer(text).type == kind


def test_answer_ignores_the_question_itself():
    assert parse_answer("Is it okay if my assistant remembers you?") is None
    assert parse_answer("Okay if I remember you?") is None
    assert parse_answer("Can it remember your face") is None


def test_answer_name_correction():
    a = parse_answer("No, it's Sara.")
    assert a.type == "correction" and a.args["name"] == "Sara"
    a = parse_answer("Actually my name is Sierra")
    assert a.type == "correction" and a.args["name"] == "Sierra"


@pytest.mark.parametrize("text,kind,args", [
    ("Who's here?", "who_here", {}),
    ("who is around", "who_here", {}),
    ("Where's my wallet?", "where_item", {"item": "wallet", "detail": False}),
    ("where are my keys", "where_item", {"item": "keys", "detail": False}),
    ("Where did I leave my AirPods?", "where_item", {"item": "airpods", "detail": False}),
    ("where did my keys go", "where_item", {"item": "keys", "detail": False}),
    ("where's my wallet in detail", "where_item", {"item": "wallet", "detail": True}),
    ("Remember this, it's my keys.", "remember_item", {"item": "keys"}),
    ("remember this as my wallet", "remember_item", {"item": "wallet"}),
    ("Remember this, it's my wallet", "remember_item", {"item": "wallet"}),
    ("this is my water bottle", "remember_item", {"item": "water bottle"}),
    ("Save this person as Sarah.", "save_person", {"name": "Sarah"}),
    ("save this person as mary jane", "save_person", {"name": "Mary"}),
    ("Forget Sarah.", "forget", {"target": "sarah"}),
    ("forget my keys", "forget", {"target": "keys"}),
    ("List people", "list_people", {}),
    ("who do you know", "list_people", {}),
    ("Repeat that.", "repeat", {}),
    ("quiet mode", "quiet_on", {}),
    ("quiet mode off", "quiet_off", {}),
    ("Yes", "yes", {}),
    ("no", "no", {}),
    ("rename Sierra to Sarah", "rename", {"old": "Sierra", "new": "Sarah", "old_raw": "sierra", "new_raw": "sarah"}),
    ("rename my wallet to my black wallet", "rename",
     {"old": None, "new": None, "old_raw": "my wallet", "new_raw": "my black wallet"}),
    ("change the name of the kitchen to the break room", "rename",
     {"old": None, "new": None, "old_raw": "the kitchen", "new_raw": "the break room"}),
    ("No, her name is Sara", "correction", {"name": "Sara"}),
    ("describe it", "describe", {"item": None}),
    ("hey, um, where's my phone please", "where_item", {"item": "phone", "detail": False}),
    ("cancel", "cancel", {}),
    ("what's the weather", "unknown", {}),
    ("this is my bedroom", "set_place", {"place": "bedroom", "mine": True}),
    ("this is the library", "set_place", {"place": "library", "mine": False}),
    ("this is the guest room", "set_place", {"place": "guest room", "mine": False}),
    ("this is Sarah's room", "set_place", {"place": "sarah's room", "mine": None}),
    ("Use left and right.", "direction_style", {"style": "words"}),
    ("use clock directions", "direction_style", {"style": "clock"}),
    ("switch to both", "direction_style", {"style": "both"}),
    ("Done.", "done", {}),
    ("that's it", "done", {}),
])
def test_commands(text, kind, args):
    intent = parse_command(text)
    assert intent.type == kind, (text, intent)
    assert intent.args == args


@pytest.mark.parametrize("text", [
    "Alright if it remembers you", "okay with you if I save your face", "Sarah, I'd like to save your face",
    "fine if it keeps your face",
])
def test_user_asking_without_question_mark_is_not_a_yes(text):
    assert parse_answer(text) is None
