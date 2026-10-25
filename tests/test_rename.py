"""Renaming items and places (people: test_storage_output / the router's existing path)."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from rig import Rig

WALLET, KEYS, TABLE = (40, 90, 200), (200, 60, 160), (60, 60, 60)
STUDY, HALL = (30, 140, 70), (120, 120, 20)
ON_TABLE = [(TABLE, "table", (200, 450, 1100, 700)), (WALLET, "wallet", (560, 470, 660, 520))]


def _learn_item(rig, name, colour):
    rig.hear(f"remember this, it's my {name}", source="ptt")
    rig.run(rig.cfg.objects.enroll_delay_s + rig.cfg.objects.enroll_s + 1.0,
            objects=[(colour, name, (560, 250, 720, 360))])
    assert rig.words()[-1] == f"Saved your {name}."


def _learn_place(rig, command, colour):
    rig.background = colour
    rig.hear(command, source="ptt")
    rig.run(rig.cfg.places.enroll_delay_s + rig.cfg.places.enroll_s + 1.0)


def test_rename_an_item_keeps_its_views_and_sightings(storage):
    rig = Rig(storage)
    _learn_item(rig, "wallet", WALLET)
    rig.run(3.0, objects=ON_TABLE)
    rig.run(1.0)
    rig.hear("rename my wallet to my black wallet", source="ptt")
    assert rig.words()[-1] == "Renamed your wallet to black wallet."
    rig.clock.advance(10 * 60)
    rig.hear("where's my black wallet?", source="ptt")
    assert rig.words()[-1] == "Your black wallet was last seen 10 minutes ago, on the table."
    rig.hear("where's my wallet?", source="ptt")
    assert rig.words()[-1] == "I don't know your wallet yet."
    rig.hear("list items", source="ptt")
    assert rig.words()[-1] == "Saved items: black wallet."
    # Still recognised under the new name.
    rig.run(3.0, objects=ON_TABLE)
    rig.hear("where's my black wallet?", source="ptt")
    assert rig.words()[-1].startswith("Your black wallet is in view")


def test_rename_an_item_to_one_that_exists_is_refused(storage):
    rig = Rig(storage)
    _learn_item(rig, "wallet", WALLET)
    _learn_item(rig, "keys", KEYS)
    rig.hear("rename wallet to keys", source="ptt")
    assert rig.words()[-1] == "You already have an item called keys. Forget one of them first."
    assert storage.find_item("wallet") is not None


def test_rename_a_place_and_its_sightings(storage):
    rig = Rig(storage)
    _learn_place(rig, "this is the study", STUDY)
    _learn_item(rig, "wallet", WALLET)
    rig.background = STUDY
    rig.run(3.0, objects=ON_TABLE)
    rig.background = HALL
    rig.run(4.0)
    rig.hear("rename the study to my office", source="ptt")
    assert rig.words()[-1] == "Renamed the study to your office."
    rig.clock.advance(5 * 60)
    rig.hear("where's my wallet?", source="ptt")
    assert rig.words()[-1] == "Your wallet was last seen 5 minutes ago, on the table in your office."
    rig.hear("list places", source="ptt")
    assert rig.words()[-1] == "Saved places: office."
    rig.background = STUDY
    rig.run(1.0)
    rig.hear("where am I?", source="ptt")
    assert rig.words()[-1] == "Probably your office."


def test_rename_this_room(storage):
    rig = Rig(storage)
    _learn_place(rig, "this is the study", STUDY)
    rig.run(1.0)
    rig.hear("rename this room to the guest room", source="ptt")
    assert rig.words()[-1] == "Renamed the study to the guest room."
    rig.background = HALL
    rig.run(1.0)
    rig.hear("rename this room to the kitchen", source="ptt")
    assert rig.words()[-1] == "I don't recognise this place."


def test_rename_a_person_still_works_and_unknown_names_are_reported(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sierra", 200)
    rig.hear("rename Sierra to Sarah", source="ptt")
    assert rig.words()[-1] == "Renamed to Sarah."
    assert storage.find_person("Sarah") is not None
    rig.hear("rename spaceship to rocket", source="ptt")
    assert rig.words()[-1] == "I don't know spaceship."


def test_manage_script_renames(storage, tmp_path):
    storage.add_item("wallet", [np.ones(512, np.float32) / np.sqrt(512)])
    storage.add_place("study", [np.ones(512, np.float32) / np.sqrt(512)])
    db = storage.path
    storage.close()
    script = Path(__file__).resolve().parents[1] / "scripts" / "manage.py"
    run = lambda *args: subprocess.run([sys.executable, str(script), "--db", db, *args],  # noqa: E731
                                       capture_output=True, text=True)
    out = run("rename", "wallet", "black wallet")
    assert out.returncode == 0 and "Renamed item wallet to black wallet" in out.stdout
    out = run("rename", "study", "my office")
    assert out.returncode == 0 and "Renamed place study to office" in out.stdout
    out = run("rename", "nothing", "x")
    assert out.returncode == 1
