"""Unit tests for ComfyUI-PromptLibrary.

Run from inside the comfy venv:
    /home/junie/comfy-env/bin/python -m pytest tests/ -v

The module imports `server` (ComfyUI's PromptServer) at module load. We stub it
before import so tests run without a live ComfyUI process.
"""

import asyncio
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
NODE_DIR = HERE.parent
INIT_PY = NODE_DIR / "__init__.py"


def _stub_server() -> None:
    fake_server = types.ModuleType("server")

    class FakeRoutes:
        def get(self, *_a, **_k):
            return lambda f: f

        def post(self, *_a, **_k):
            return lambda f: f

    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=FakeRoutes())
    )
    sys.modules["server"] = fake_server


def _load_module(tmp_root: Path):
    """Re-import the package against a fresh temp data dir."""
    _stub_server()
    spec = importlib.util.spec_from_file_location(f"plib_{tmp_root.name}", INIT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect storage paths to the temp dir.
    mod.DATA_DIR = tmp_root / "data"
    mod.IMAGES_DIR = mod.DATA_DIR / "images"
    mod.STORE_PATH = mod.DATA_DIR / "prompts.json"
    mod.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return mod


class FakeFileField:
    """Mimics aiohttp's FileField (the .file attribute and .filename)."""
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = io.BytesIO(data)


class FakeRequest:
    def __init__(self, *, post_data=None, json_data=None, match_info=None):
        self._post_data = post_data or {}
        self._json_data = json_data or {}
        self.match_info = match_info or {}

    async def post(self):
        return self._post_data

    async def json(self):
        return self._json_data


class PromptLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plib_test_"))
        self.mod = _load_module(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- store helpers --------------------------------------------------

    def test_load_returns_empty_when_no_store(self):
        self.assertEqual(self.mod._load(), [])

    def test_save_then_load_roundtrip(self):
        items = [{"id": "abc", "name": "n", "text": "t"}]
        self.mod._save(items)
        self.assertEqual(self.mod._load(), items)
        self.assertTrue(self.mod.STORE_PATH.exists())

    def test_load_handles_corrupted_json(self):
        self.mod.STORE_PATH.write_text("not json {{{")
        self.assertEqual(self.mod._load(), [])

    def test_load_handles_non_list_json(self):
        self.mod.STORE_PATH.write_text('{"foo": "bar"}')
        self.assertEqual(self.mod._load(), [])

    def test_safe_id_accepts_valid(self):
        for ok in ["abc", "a_b-c", "ABC123", "x" * 64]:
            self.assertEqual(self.mod._safe_id(ok), ok)

    def test_safe_id_rejects_invalid(self):
        for bad in ["", "..", "a/b", "with space", "x" * 65, "café", None and ""]:
            self.assertIsNone(self.mod._safe_id(bad or ""))

    # ---- node logic -----------------------------------------------------

    def test_load_prompt_returns_text_for_known_id(self):
        self.mod._save([{"id": "k1", "name": "n", "text": "hello"}])
        node = self.mod.PromptLibrary()
        self.assertEqual(node.load_prompt("k1"), ("hello",))

    def test_load_prompt_empty_for_missing_id(self):
        node = self.mod.PromptLibrary()
        self.assertEqual(node.load_prompt("missing"), ("",))

    def test_load_prompt_empty_for_blank_id(self):
        node = self.mod.PromptLibrary()
        self.assertEqual(node.load_prompt(""), ("",))

    def test_is_changed_reflects_text(self):
        self.mod._save([{"id": "k", "name": "n", "text": "v1"}])
        self.assertEqual(self.mod.PromptLibrary.IS_CHANGED("k"), "v1")
        self.mod._save([{"id": "k", "name": "n", "text": "v2"}])
        self.assertEqual(self.mod.PromptLibrary.IS_CHANGED("k"), "v2")

    def test_input_types_shape(self):
        spec = self.mod.PromptLibrary.INPUT_TYPES()
        self.assertIn("required", spec)
        self.assertIn("prompt_id", spec["required"])
        self.assertEqual(spec["required"]["prompt_id"][0], "STRING")

    # ---- routes ---------------------------------------------------------

    def test_list_route_empty(self):
        resp = asyncio.run(self.mod.list_prompts(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(body, {"prompts": []})

    def test_upsert_creates_new_prompt(self):
        req = FakeRequest(post_data={"name": "Cat", "text": "fluffy"})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        body = json.loads(resp.body)
        self.assertEqual(body["name"], "Cat")
        self.assertEqual(body["text"], "fluffy")
        self.assertFalse(body["has_image"])
        self.assertTrue(body["id"])
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], body["id"])

    def test_upsert_requires_name(self):
        req = FakeRequest(post_data={"name": "  ", "text": "x"})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)
        self.assertIn(b"name required", resp.body)

    def test_upsert_with_image_writes_file(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # not a valid PNG, just a payload
        req = FakeRequest(post_data={
            "name": "Tiger",
            "text": "stripey",
            "image": FakeFileField("tiger.PNG", png_bytes),
        })
        resp = asyncio.run(self.mod.upsert_prompt(req))
        body = json.loads(resp.body)
        self.assertTrue(body["has_image"])
        path = self.mod._image_path_for(body["id"])
        self.assertIsNotNone(path)
        self.assertEqual(path.suffix, ".png")
        self.assertEqual(path.read_bytes(), png_bytes)

    def test_upsert_rejects_bad_extension(self):
        req = FakeRequest(post_data={
            "name": "Bad",
            "text": "",
            "image": FakeFileField("evil.exe", b"MZ\x00\x00"),
        })
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)
        self.assertIn(b"unsupported image type", resp.body)

    def test_upsert_rejects_oversized_image(self):
        big = b"\x00" * (self.mod._MAX_IMAGE_BYTES + 1)
        req = FakeRequest(post_data={
            "name": "Huge",
            "text": "",
            "image": FakeFileField("huge.png", big),
        })
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)
        self.assertIn(b"too large", resp.body)

    def test_upsert_update_existing_keeps_id(self):
        req1 = FakeRequest(post_data={"name": "A", "text": "v1"})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body1["id"]
        req2 = FakeRequest(post_data={"id": pid, "name": "A", "text": "v2"})
        body2 = json.loads(asyncio.run(self.mod.upsert_prompt(req2)).body)
        self.assertEqual(body2["id"], pid)
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "v2")

    def test_upsert_clear_image_removes_file(self):
        req1 = FakeRequest(post_data={
            "name": "X", "text": "",
            "image": FakeFileField("a.png", b"data"),
        })
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body1["id"]
        self.assertTrue(self.mod._image_path_for(pid))
        req2 = FakeRequest(post_data={"id": pid, "name": "X", "text": "", "clear_image": "1"})
        body2 = json.loads(asyncio.run(self.mod.upsert_prompt(req2)).body)
        self.assertFalse(body2["has_image"])
        self.assertIsNone(self.mod._image_path_for(pid))

    def test_upsert_with_image_replaces_existing(self):
        # First upload .png, then upload .jpg — old png file should be removed.
        req1 = FakeRequest(post_data={
            "name": "Y", "text": "",
            "image": FakeFileField("a.png", b"png-data"),
        })
        pid = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)["id"]
        png_path = self.mod._image_path_for(pid)
        self.assertEqual(png_path.suffix, ".png")
        req2 = FakeRequest(post_data={
            "id": pid, "name": "Y", "text": "",
            "image": FakeFileField("a.JPG", b"jpg-data"),
        })
        asyncio.run(self.mod.upsert_prompt(req2))
        self.assertFalse(png_path.exists())
        new_path = self.mod._image_path_for(pid)
        self.assertEqual(new_path.suffix, ".jpg")
        self.assertEqual(new_path.read_bytes(), b"jpg-data")

    def test_upsert_rejects_invalid_id(self):
        req = FakeRequest(post_data={"id": "../etc", "name": "x", "text": ""})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)
        self.assertIn(b"invalid id", resp.body)

    def test_delete_route_removes_entry_and_image(self):
        req1 = FakeRequest(post_data={
            "name": "Z", "text": "",
            "image": FakeFileField("a.png", b"data"),
        })
        pid = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)["id"]
        self.assertTrue(self.mod._image_path_for(pid))

        del_req = FakeRequest(json_data={"id": pid})
        resp = asyncio.run(self.mod.delete_prompt(del_req))
        self.assertEqual(json.loads(resp.body), {"ok": True})
        self.assertEqual(self.mod._load(), [])
        self.assertIsNone(self.mod._image_path_for(pid))

    def test_delete_route_rejects_invalid_id(self):
        req = FakeRequest(json_data={"id": "../etc"})
        resp = asyncio.run(self.mod.delete_prompt(req))
        self.assertEqual(resp.status, 400)

    def test_image_route_404_when_missing(self):
        req = FakeRequest(match_info={"prompt_id": "nope"})
        resp = asyncio.run(self.mod.get_image(req))
        self.assertEqual(resp.status, 404)

    def test_image_route_400_on_bad_id(self):
        req = FakeRequest(match_info={"prompt_id": "../etc"})
        resp = asyncio.run(self.mod.get_image(req))
        self.assertEqual(resp.status, 400)

    def test_node_class_mappings_exposed(self):
        self.assertIn("PromptLibrary", self.mod.NODE_CLASS_MAPPINGS)
        self.assertIn("PromptLibrarySave", self.mod.NODE_CLASS_MAPPINGS)
        self.assertEqual(self.mod.WEB_DIRECTORY, "./web")

    def test_version_exposed(self):
        self.assertRegex(self.mod.__version__, r"^\d+\.\d+\.\d+$")

    # ---- save node ------------------------------------------------------

    def _fake_image(self, h=4, w=4):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")
        return np.full((1, h, w, 3), 0.5, dtype=np.float32)

    def test_save_node_input_shape(self):
        spec = self.mod.PromptLibrarySave.INPUT_TYPES()
        self.assertIn("name", spec["required"])
        self.assertIn("text", spec["required"])
        self.assertIn("thumbnail", spec["optional"])
        self.assertEqual(self.mod.PromptLibrarySave.RETURN_TYPES, ("STRING", "STRING"))
        self.assertTrue(self.mod.PromptLibrarySave.OUTPUT_NODE)

    def test_save_node_creates_entry(self):
        node = self.mod.PromptLibrarySave()
        text, pid = node.save(name="From Workflow", text="generated prompt")
        self.assertTrue(pid)
        self.assertEqual(text, "generated prompt")
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "From Workflow")
        self.assertEqual(items[0]["text"], "generated prompt")

    def test_save_node_requires_name(self):
        node = self.mod.PromptLibrarySave()
        with self.assertRaises(ValueError):
            node.save(name="  ", text="x")

    def test_save_node_with_thumbnail_writes_png(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("PIL/numpy not available")
        node = self.mod.PromptLibrarySave()
        img = self._fake_image()
        _, pid = node.save(name="ThumbTest", text="t", thumbnail=img)
        path = self.mod._image_path_for(pid)
        self.assertIsNotNone(path)
        self.assertEqual(path.suffix, ".png")
        self.assertGreater(path.stat().st_size, 0)

    def test_save_node_updates_existing_by_id(self):
        node = self.mod.PromptLibrarySave()
        _, pid = node.save(name="A", text="v1")
        _, pid2 = node.save(name="A", text="v2", prompt_id=pid)
        self.assertEqual(pid, pid2)
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "v2")

    def test_save_node_overwrite_by_name_replaces(self):
        node = self.mod.PromptLibrarySave()
        node.save(name="Same Name", text="v1")
        node.save(name="Same Name", text="v2", overwrite_by_name=True)
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "v2")

    def test_save_node_no_overwrite_creates_second_entry(self):
        node = self.mod.PromptLibrarySave()
        node.save(name="Same Name", text="v1")
        node.save(name="Same Name", text="v2")  # overwrite_by_name defaults to False
        items = self.mod._load()
        self.assertEqual(len(items), 2)

    def test_save_node_rejects_invalid_prompt_id(self):
        node = self.mod.PromptLibrarySave()
        with self.assertRaises(ValueError):
            node.save(name="X", text="t", prompt_id="../etc")

    # ---- slugify / unique id -------------------------------------------

    def test_slugify_basic(self):
        cases = {
            "Cyberpunk Style": "cyberpunk_style",
            "Cyberpunk Style 2!": "cyberpunk_style_2",
            "  spaces  ": "spaces",
            "Hello---World": "hello_world",
            "café": "caf",
            "": "",
            "!!!": "",
        }
        for inp, expected in cases.items():
            self.assertEqual(self.mod._slugify(inp), expected, f"input={inp!r}")

    def test_slugify_truncates_to_64(self):
        self.assertEqual(len(self.mod._slugify("a" * 200)), 64)

    def test_unique_id_no_collision(self):
        self.assertEqual(self.mod._unique_id("foo", set()), "foo")

    def test_unique_id_with_collision(self):
        self.assertEqual(self.mod._unique_id("foo", {"foo"}), "foo_2")
        self.assertEqual(self.mod._unique_id("foo", {"foo", "foo_2"}), "foo_3")

    def test_unique_id_empty_falls_back_to_uuid(self):
        result = self.mod._unique_id("", set())
        self.assertRegex(result, r"^[a-f0-9]{12}$")

    def test_save_node_uses_slugified_name_as_id(self):
        node = self.mod.PromptLibrarySave()
        _, pid = node.save(name="Cyberpunk Style", text="t")
        self.assertEqual(pid, "cyberpunk_style")

    def test_save_node_collision_appends_suffix(self):
        node = self.mod.PromptLibrarySave()
        _, pid1 = node.save(name="Same Name", text="v1")
        _, pid2 = node.save(name="Same Name", text="v2")
        self.assertEqual(pid1, "same_name")
        self.assertEqual(pid2, "same_name_2")

    def test_upsert_uses_slugified_id_when_unspecified(self):
        req = FakeRequest(post_data={"name": "Movie Poster", "text": "x"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        self.assertEqual(body["id"], "movie_poster")

    def test_upsert_collision_appends_suffix(self):
        req1 = FakeRequest(post_data={"name": "Foo", "text": "a"})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        req2 = FakeRequest(post_data={"name": "Foo", "text": "b"})
        body2 = json.loads(asyncio.run(self.mod.upsert_prompt(req2)).body)
        self.assertEqual(body1["id"], "foo")
        self.assertEqual(body2["id"], "foo_2")

    def test_upsert_all_symbol_name_falls_back_to_uuid(self):
        req = FakeRequest(post_data={"name": "!!!", "text": "x"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        self.assertRegex(body["id"], r"^[a-f0-9]{12}$")

    # ---- tags ----------------------------------------------------------

    def test_parse_tags_string_normalizes(self):
        self.assertEqual(self.mod._parse_tags("Character, Fantasy, character"),
                         ["character", "fantasy"])

    def test_parse_tags_list(self):
        self.assertEqual(self.mod._parse_tags(["Sci-Fi", "Sci-Fi", "  noir "]),
                         ["sci-fi", "noir"])

    def test_parse_tags_none_or_empty(self):
        self.assertEqual(self.mod._parse_tags(None), [])
        self.assertEqual(self.mod._parse_tags(""), [])
        self.assertEqual(self.mod._parse_tags(",,, ,"), [])

    def test_upsert_persists_tags(self):
        req = FakeRequest(post_data={"name": "Knight", "text": "armor", "tags": "character, fantasy"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        self.assertEqual(body["tags"], ["character", "fantasy"])
        items = self.mod._load()
        self.assertEqual(items[0]["tags"], ["character", "fantasy"])

    def test_list_route_returns_tags(self):
        req1 = FakeRequest(post_data={"name": "Knight", "text": "x", "tags": "character"})
        asyncio.run(self.mod.upsert_prompt(req1))
        resp = asyncio.run(self.mod.list_prompts(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(body["prompts"][0]["tags"], ["character"])

    def test_tags_route_returns_unique_sorted(self):
        for tags in ["fantasy, character", "character, sci-fi", "noir"]:
            req = FakeRequest(post_data={"name": f"P_{tags}", "text": "x", "tags": tags})
            asyncio.run(self.mod.upsert_prompt(req))
        resp = asyncio.run(self.mod.list_tags(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(body["tags"], ["character", "fantasy", "noir", "sci-fi"])

    def test_save_node_persists_tags(self):
        node = self.mod.PromptLibrarySave()
        _, pid = node.save(name="Wizard", text="staff", tags="Character, Fantasy")
        items = self.mod._load()
        entry = next(i for i in items if i["id"] == pid)
        self.assertEqual(entry["tags"], ["character", "fantasy"])

    # ---- timestamps ----------------------------------------------------

    def test_upsert_records_timestamps(self):
        req = FakeRequest(post_data={"name": "TS", "text": "x"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        items = self.mod._load()
        entry = next(i for i in items if i["id"] == body["id"])
        self.assertIn("created_at", entry)
        self.assertIn("updated_at", entry)
        self.assertEqual(entry["created_at"], entry["updated_at"])

    def test_upsert_update_changes_only_updated_at(self):
        req1 = FakeRequest(post_data={"name": "TS", "text": "v1"})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body1["id"]
        items = self.mod._load()
        created_at = next(i for i in items if i["id"] == pid)["created_at"]
        # Sleep to let the clock advance past the resolution.
        import time as _t
        _t.sleep(0.01)
        req2 = FakeRequest(post_data={"id": pid, "name": "TS", "text": "v2"})
        asyncio.run(self.mod.upsert_prompt(req2))
        entry = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertEqual(entry["created_at"], created_at)
        self.assertGreater(entry["updated_at"], created_at)

    def test_list_returns_timestamps(self):
        req = FakeRequest(post_data={"name": "TS", "text": "x"})
        asyncio.run(self.mod.upsert_prompt(req))
        body = json.loads(asyncio.run(self.mod.list_prompts(FakeRequest())).body)
        self.assertGreater(body["prompts"][0]["created_at"], 0)
        self.assertGreater(body["prompts"][0]["updated_at"], 0)

    # ---- CSV import ----------------------------------------------------

    def test_import_csv_basic(self):
        text = "name,text,tags,id\nKnight,armor,character;fantasy,knight\nMage,staff,character;magic,\n"
        added, updated, errors = self.mod._import_csv(text)
        self.assertEqual(added, 2)
        self.assertEqual(updated, 0)
        self.assertEqual(errors, [])
        items = self.mod._load()
        self.assertEqual(len(items), 2)
        knight = next(i for i in items if i["id"] == "knight")
        self.assertEqual(knight["text"], "armor")
        self.assertEqual(knight["tags"], ["character", "fantasy"])
        mage = next(i for i in items if i["name"] == "Mage")
        self.assertEqual(mage["id"], "mage")  # auto-slugified

    def test_import_csv_updates_existing(self):
        # First add via upsert
        req = FakeRequest(post_data={"name": "Knight", "text": "v1"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        pid = body["id"]
        # Then import a CSV that updates the same id
        text = f"name,text,tags,id\nKnight,v2,character,{pid}\n"
        added, updated, errors = self.mod._import_csv(text)
        self.assertEqual(added, 0)
        self.assertEqual(updated, 1)
        items = self.mod._load()
        self.assertEqual(items[0]["text"], "v2")

    def test_import_csv_skips_invalid_rows(self):
        text = "name,text,tags,id\n,nothing,,\nGood,t,,\nBad,t,,../etc\n"
        added, _, errors = self.mod._import_csv(text)
        self.assertEqual(added, 1)
        self.assertEqual(len(errors), 2)

    def test_import_csv_rejects_missing_name_column(self):
        text = "title,text\nfoo,bar\n"
        _, _, errors = self.mod._import_csv(text)
        self.assertTrue(any("name" in e for e in errors))

    def test_import_csv_route(self):
        text = "name,text,tags,id\nFoo,bar,t1;t2,foo\n"
        req = FakeRequest(post_data={"csv": text})
        resp = asyncio.run(self.mod.import_csv_route(req))
        body = json.loads(resp.body)
        self.assertEqual(body["added"], 1)
        self.assertEqual(self.mod._load()[0]["tags"], ["t1", "t2"])

    # ---- versioning ----------------------------------------------------

    def test_upsert_first_create_no_history(self):
        req = FakeRequest(post_data={"name": "V", "text": "v1"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        item = next(i for i in self.mod._load() if i["id"] == body["id"])
        self.assertEqual(item.get("history", []), [])

    def test_upsert_update_pushes_history(self):
        req1 = FakeRequest(post_data={"name": "V", "text": "v1"})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body1["id"]
        req2 = FakeRequest(post_data={"id": pid, "name": "V", "text": "v2"})
        asyncio.run(self.mod.upsert_prompt(req2))
        item = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertEqual(len(item["history"]), 1)
        self.assertEqual(item["history"][0]["text"], "v1")

    def test_upsert_noop_doesnt_push_history(self):
        req = FakeRequest(post_data={"name": "V", "text": "v1"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        pid = body["id"]
        # identical second call: no history bump
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"id": pid, "name": "V", "text": "v1"})))
        item = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertEqual(len(item.get("history", [])), 0)

    def test_history_caps_at_20(self):
        item = {"id": "x", "name": "a", "text": "0", "tags": []}
        for i in range(25):
            item["text"] = str(i)
            self.mod._push_history(item)
        self.assertEqual(len(item["history"]), 20)
        # Oldest 5 dropped: history should now start at "5"
        self.assertEqual(item["history"][0]["text"], "5")
        self.assertEqual(item["history"][-1]["text"], "24")

    def test_history_route(self):
        req1 = FakeRequest(post_data={"name": "V", "text": "v1"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body["id"]
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"id": pid, "name": "V", "text": "v2"})))
        resp = asyncio.run(self.mod.get_history(FakeRequest(match_info={"prompt_id": pid})))
        data = json.loads(resp.body)
        self.assertEqual(len(data["history"]), 1)
        self.assertEqual(data["history"][0]["text"], "v1")

    def test_revert_applies_snapshot(self):
        # Create v1, edit to v2, revert to v1.
        req1 = FakeRequest(post_data={"name": "V", "text": "v1", "tags": "a"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body["id"]
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"id": pid, "name": "V2", "text": "v2", "tags": "b"})))
        item_after_v2 = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertEqual(item_after_v2["text"], "v2")
        snap_ts = item_after_v2["history"][0]["ts"]

        resp = asyncio.run(self.mod.revert_prompt(FakeRequest(json_data={"id": pid, "ts": snap_ts})))
        data = json.loads(resp.body)
        self.assertEqual(data["text"], "v1")
        self.assertEqual(data["name"], "V")
        self.assertEqual(data["tags"], ["a"])

        # Revert itself should be undoable: a new history entry should be there with v2.
        item = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertGreaterEqual(len(item["history"]), 2)
        self.assertEqual(item["history"][-1]["text"], "v2")

    def test_revert_404_for_unknown_id(self):
        resp = asyncio.run(self.mod.revert_prompt(FakeRequest(json_data={"id": "nope", "ts": 1.0})))
        self.assertEqual(resp.status, 404)

    def test_revert_404_for_unknown_ts(self):
        req = FakeRequest(post_data={"name": "V", "text": "v1"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        resp = asyncio.run(self.mod.revert_prompt(FakeRequest(json_data={"id": body["id"], "ts": 9999999})))
        self.assertEqual(resp.status, 404)

    def test_history_route_404_for_unknown_id(self):
        resp = asyncio.run(self.mod.get_history(FakeRequest(match_info={"prompt_id": "nope"})))
        self.assertEqual(resp.status, 404)

    # ---- watcher (smoke) -----------------------------------------------

    def test_watcher_thread_started(self):
        # _start_watcher runs once at module import; check the thread exists.
        names = [t.name for t in __import__("threading").enumerate()]
        self.assertIn("prompt-library-watcher", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
