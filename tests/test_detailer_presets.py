"""Unit tests for the Smart Detailer named-preset store."""

import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
NODE_DIR = HERE.parent


class DetailerPresetsTests(unittest.TestCase):
    def setUp(self):
        # Re-import the module against a temp data dir so concurrent test
        # runs and the real repo's data/ are untouched.
        self.tmp = Path(tempfile.mkdtemp(prefix="grm_det_presets_"))
        if str(NODE_DIR) not in sys.path:
            sys.path.insert(0, str(NODE_DIR))
        # Force a fresh import so module-level paths can be redirected to
        # the tmpdir before any presets file is written.
        sys.modules.pop("detailer_presets", None)
        self.mod = importlib.import_module("detailer_presets")
        self.mod.DETAILER_PRESETS_PATH = self.tmp / "data" / "detailer_presets.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        sys.modules.pop("detailer_presets", None)

    # ---- seeding --------------------------------------------------------

    def test_seed_writes_bundled_presets_when_file_missing(self):
        self.assertFalse(self.mod.DETAILER_PRESETS_PATH.exists())
        self.mod.seed_default_presets()
        self.assertTrue(self.mod.DETAILER_PRESETS_PATH.exists())
        presets = self.mod.list_presets()
        names = [p["name"] for p in presets]
        # All bundled seed presets must land
        for expected in ("Default (reset)", "Portrait Hero", "Group / Crowd",
                         "Anime / Illustration", "Photo Realistic",
                         "Hands Focus", "Quick / Speed"):
            self.assertIn(expected, names)
        # Builtins flagged
        builtins = [p for p in presets if p["builtin"]]
        self.assertEqual(len(builtins), len(presets))

    def test_seed_is_idempotent_when_file_exists(self):
        # First seed runs.
        self.mod.seed_default_presets()
        first = self.mod.list_presets()
        # Delete one builtin, re-seed should NOT restore it (sticky deletion).
        target = first[0]
        self.mod.delete_preset(target["id"])
        self.mod.seed_default_presets()
        remaining = self.mod.list_presets()
        self.assertEqual(len(remaining), len(first) - 1)
        self.assertNotIn(target["id"], [p["id"] for p in remaining])

    # ---- upsert / get / delete -----------------------------------------

    def test_upsert_creates_new_preset(self):
        saved = self.mod.upsert_preset("My Test", {"enable_face": True, "steps": 30})
        self.assertEqual(saved["name"], "My Test")
        self.assertFalse(saved["builtin"])
        self.assertEqual(saved["settings"], {"enable_face": True, "steps": 30})
        # Round-trip through list+get
        listed = self.mod.list_presets()
        self.assertEqual(len(listed), 1)
        full = self.mod.get_preset(saved["id"])
        self.assertIsNotNone(full)
        self.assertEqual(full["settings"], {"enable_face": True, "steps": 30})

    def test_upsert_with_id_updates_in_place(self):
        a = self.mod.upsert_preset("A", {"steps": 20})
        # Same id → update name + settings; created_at preserved
        b = self.mod.upsert_preset("A renamed", {"steps": 30, "cfg": 5},
                                   preset_id=a["id"])
        self.assertEqual(b["id"], a["id"])
        self.assertEqual(b["name"], "A renamed")
        self.assertEqual(b["settings"], {"steps": 30, "cfg": 5})
        self.assertEqual(b["created_at"], a["created_at"])
        # Only one preset in the store
        self.assertEqual(len(self.mod.list_presets()), 1)

    def test_upsert_by_name_match_is_case_insensitive(self):
        a = self.mod.upsert_preset("photo", {"cfg": 7})
        b = self.mod.upsert_preset("PHOTO", {"cfg": 8})
        # Same id retained because name (case-insensitive) matched.
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(len(self.mod.list_presets()), 1)
        full = self.mod.get_preset(a["id"])
        self.assertEqual(full["settings"], {"cfg": 8})

    def test_upsert_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            self.mod.upsert_preset("", {"steps": 1})
        with self.assertRaises(ValueError):
            self.mod.upsert_preset("   ", {"steps": 1})

    def test_upsert_rejects_non_dict_settings(self):
        with self.assertRaises(ValueError):
            self.mod.upsert_preset("oops", "not a dict")  # type: ignore[arg-type]

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.mod.get_preset(""))
        self.assertIsNone(self.mod.get_preset("nope"))

    def test_delete_removes_and_returns_true(self):
        a = self.mod.upsert_preset("A", {"steps": 1})
        self.assertTrue(self.mod.delete_preset(a["id"]))
        self.assertIsNone(self.mod.get_preset(a["id"]))
        self.assertEqual(self.mod.list_presets(), [])

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.mod.delete_preset("nope"))
        self.assertFalse(self.mod.delete_preset(""))

    def test_delete_then_seed_does_not_resurrect(self):
        # Specifically: seed creates the file; delete every entry; re-seed.
        # File still exists, so re-seeding is a no-op even though list is empty.
        self.mod.seed_default_presets()
        for p in list(self.mod.list_presets()):
            self.mod.delete_preset(p["id"])
        self.assertEqual(self.mod.list_presets(), [])
        self.mod.seed_default_presets()
        self.assertEqual(self.mod.list_presets(), [])

    # ---- listing / sorting ---------------------------------------------

    def test_list_sorts_builtins_first_then_alpha(self):
        self.mod.seed_default_presets()
        self.mod.upsert_preset("Zeta user", {"x": 1})
        self.mod.upsert_preset("Alpha user", {"x": 1})
        listed = self.mod.list_presets()
        # All builtins (sorted alpha among themselves) before any non-builtin.
        builtins_seen = True
        for p in listed:
            if not p["builtin"]:
                builtins_seen = False
            if not builtins_seen and p["builtin"]:
                self.fail("builtin appeared after non-builtin in sorted list")
        # User entries appear in alpha order.
        user_names = [p["name"] for p in listed if not p["builtin"]]
        self.assertEqual(user_names, ["Alpha user", "Zeta user"])

    # ---- corruption / recovery -----------------------------------------

    def test_load_treats_corrupt_file_as_empty(self):
        self.mod.DETAILER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.mod.DETAILER_PRESETS_PATH.write_text("not json {{{")
        self.assertEqual(self.mod.list_presets(), [])
        # Recovery path: a fresh upsert overwrites the broken file with a
        # valid store.
        saved = self.mod.upsert_preset("recovered", {"steps": 1})
        self.assertEqual(self.mod.get_preset(saved["id"])["name"], "recovered")

    def test_load_treats_wrong_shape_as_empty(self):
        self.mod.DETAILER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.mod.DETAILER_PRESETS_PATH.write_text(json.dumps(["a", "b"]))
        self.assertEqual(self.mod.list_presets(), [])

    def test_coerce_skips_entries_without_name(self):
        # Hand-craft a malformed entry in the store and confirm it's filtered.
        self.mod.DETAILER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.mod.DETAILER_PRESETS_PATH.write_text(json.dumps({
            "schema_version": 1,
            "presets": [
                {"id": "x", "name": "", "settings": {}},
                {"id": "y", "name": "Good", "settings": {"steps": 1}},
                "not a dict",
            ],
        }))
        listed = self.mod.list_presets()
        self.assertEqual([p["name"] for p in listed], ["Good"])


if __name__ == "__main__":
    unittest.main()
