"""Tests for the Prompt Builder tag-pack importer.

Run from inside the comfy venv:
    /home/junie/comfy-env/bin/python -m pytest tests/test_tag_pack_import.py -v
"""

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from test_prompt_library import _load_module


def _build_pb_zip(files: dict[str, list[dict]]) -> bytes:
    """Build a Prompt Builder-style zip with arbitrary Tags-*.json members."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, entries in files.items():
            zf.writestr(name, json.dumps(entries, ensure_ascii=False))
    return buf.getvalue()


class TagPackImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plib_tp_"))
        self.mod = _load_module(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_basic_import_creates_entries_with_tags(self):
        zip_bytes = _build_pb_zip({
            "Prompt Builder/Tags-Clothing.json": [
                {"name": "Anklet", "category": "Accessories/Objects", "prompt": "(anklet)"},
                {"name": "Bikini",  "category": "Clothing", "prompt": "wearing bikini"},
            ],
        })
        added, updated, errors = self.mod._import_tag_pack_zip(zip_bytes)
        self.assertEqual((added, updated, errors), (2, 0, []))

        items = self.mod._load()
        self.assertEqual(len(items), 2)
        anklet = next(i for i in items if i["name"] == "Anklet")
        self.assertEqual(anklet["text"], "(anklet)")
        self.assertIn("prompt-builder", anklet["tags"])
        self.assertIn("clothing", anklet["tags"])  # file_stem
        self.assertIn("accessories_objects", anklet["tags"])  # category slug

    def test_skips_master_custom_deleted_and_backup_dir(self):
        zip_bytes = _build_pb_zip({
            "Prompt Builder/Tags-_Master_Filtered v3.json": [
                {"name": "Master", "category": "X", "prompt": "p"},
            ],
            "Prompt Builder/Tags-Custom.json": [
                {"name": "Custom", "category": "X", "prompt": "p"},
            ],
            "Prompt Builder/Tags-Deleted.json": [
                {"name": "Deleted", "category": "X", "prompt": "p"},
            ],
            "Prompt Builder/_Original Files (Backup)/Tags-Body.json": [
                {"name": "Backup", "category": "Body", "prompt": "p"},
            ],
            "Prompt Builder/Tags-View.json": [
                {"name": "Front", "category": "View", "prompt": "front view"},
            ],
        })
        added, _, errors = self.mod._import_tag_pack_zip(zip_bytes)
        self.assertEqual(added, 1)
        self.assertEqual(errors, [])
        names = {i["name"] for i in self.mod._load()}
        self.assertEqual(names, {"Front"})

    def test_anime_and_negative_get_extra_tags(self):
        zip_bytes = _build_pb_zip({
            "Tags-AnimeBase.json": [{"name": "Catgirl", "category": "Anime Base", "prompt": "catgirl"}],
            "Tags-NegAdvancedStyle.json": [{"name": "Bad Hands", "category": "Negative", "prompt": "bad hands"}],
            "Tags-MenBase.json": [{"name": "Beard", "category": "Men Base", "prompt": "beard"}],
        })
        added, _, errors = self.mod._import_tag_pack_zip(zip_bytes)
        self.assertEqual(added, 3)
        self.assertEqual(errors, [])
        by_name = {i["name"]: i for i in self.mod._load()}
        self.assertIn("anime", by_name["Catgirl"]["tags"])
        self.assertIn("negative", by_name["Bad Hands"]["tags"])
        self.assertIn("men", by_name["Beard"]["tags"])

    def test_collisions_get_suffixed_ids(self):
        zip_bytes = _build_pb_zip({
            "Tags-Action.json": [
                {"name": "Legs Up", "category": "Action", "prompt": "(legs up:1.58)"},
                {"name": "Legs Up", "category": "Action", "prompt": "lying back, (legs up)"},
                {"name": "Legs Up", "category": "Action", "prompt": "anatomically correct"},
            ],
        })
        added, _, _ = self.mod._import_tag_pack_zip(zip_bytes)
        self.assertEqual(added, 3)
        ids = sorted(i["id"] for i in self.mod._load())
        self.assertEqual(ids, ["legs_up", "legs_up_2", "legs_up_3"])

    def test_skips_entries_missing_name_or_prompt(self):
        zip_bytes = _build_pb_zip({
            "Tags-Body.json": [
                {"name": "", "category": "Body", "prompt": "p"},
                {"name": "Valid", "category": "Body", "prompt": ""},
                {"name": "Good", "category": "Body", "prompt": "real"},
            ],
        })
        added, _, _ = self.mod._import_tag_pack_zip(zip_bytes)
        self.assertEqual(added, 1)
        self.assertEqual([i["name"] for i in self.mod._load()], ["Good"])

    def test_bad_zip_returns_error(self):
        added, updated, errors = self.mod._import_tag_pack_zip(b"not a zip")
        self.assertEqual((added, updated), (0, 0))
        self.assertTrue(errors)

    def test_non_list_json_records_error(self):
        zip_bytes = _build_pb_zip({
            "Tags-Style.json": [{"name": "ok", "category": "Style", "prompt": "x"}],
        })
        # Re-pack with one file replaced by a non-list value.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Tags-Body.json", json.dumps({"oops": True}))
            zf.writestr("Tags-Style.json", json.dumps([
                {"name": "ok", "category": "Style", "prompt": "x"},
            ]))
        added, _, errors = self.mod._import_tag_pack_zip(buf.getvalue())
        self.assertEqual(added, 1)
        self.assertTrue(any("not a list" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
