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
    # Redirect storage paths to the temp dir so tests don't pollute the real repo.
    mod.ROOT = tmp_root
    mod.DATA_DIR = tmp_root / "data"
    mod.IMAGES_DIR = mod.DATA_DIR / "images"
    mod.STORE_PATH = mod.DATA_DIR / "prompts.json"
    mod.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return mod


def _real_png(size=(8, 8), color=(200, 100, 50)) -> bytes:
    """Build an actually-decodable PNG so the upsert route's thumbnail step
    (which now PIL-decodes and downscales) doesn't reject the test payload."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


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

    def test_load_prompt_joins_multiple_ids_with_separator(self):
        self.mod._save([
            {"id": "a", "name": "A", "text": "alpha"},
            {"id": "b", "name": "B", "text": "beta"},
            {"id": "c", "name": "C", "text": "gamma"},
        ])
        node = self.mod.PromptLibrary()
        self.assertEqual(node.load_prompt("a,b,c"), ("alpha, beta, gamma",))
        self.assertEqual(node.load_prompt("a, b , c"), ("alpha, beta, gamma",))
        self.assertEqual(node.load_prompt("a,c", separator=" | "), ("alpha | gamma",))

    def test_load_prompt_skips_missing_in_multi_id(self):
        self.mod._save([{"id": "a", "name": "A", "text": "alpha"}])
        node = self.mod.PromptLibrary()
        self.assertEqual(node.load_prompt("a,missing,a"), ("alpha, alpha",))

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

    def test_upsert_downscales_oversized_image(self):
        # 4k input -> output max edge should match _THUMBNAIL_MAX_EDGE (512).
        from PIL import Image
        big = io.BytesIO()
        Image.new("RGB", (4096, 4096), (50, 60, 70)).save(big, format="PNG")
        big_bytes = big.getvalue()
        req = FakeRequest(post_data={
            "name": "Big", "text": "",
            "image": FakeFileField("big.png", big_bytes),
        })
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        path = self.mod._image_path_for(body["id"])
        self.assertIsNotNone(path)
        with Image.open(path) as out:
            self.assertLessEqual(max(out.size), self.mod._THUMBNAIL_MAX_EDGE)
        self.assertLess(path.stat().st_size, len(big_bytes))

    def test_upsert_with_image_writes_file(self):
        png_bytes = _real_png()
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
        # Opaque RGB images are saved as JPEG (smaller than PNG); we don't keep
        # the source bytes — the file is downscaled and re-encoded.
        self.assertEqual(path.suffix, ".jpg")
        self.assertGreater(path.stat().st_size, 0)

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
            "image": FakeFileField("a.png", _real_png()),
        })
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        pid = body1["id"]
        self.assertTrue(self.mod._image_path_for(pid))
        req2 = FakeRequest(post_data={"id": pid, "name": "X", "text": "", "clear_image": "1"})
        body2 = json.loads(asyncio.run(self.mod.upsert_prompt(req2)).body)
        self.assertFalse(body2["has_image"])
        self.assertIsNone(self.mod._image_path_for(pid))

    def test_upsert_with_image_replaces_existing(self):
        # Upload an opaque image (saved as .jpg by the resizer) then an image
        # with alpha (saved as .png) — the old .jpg should be removed when the
        # extension flips.
        from PIL import Image
        rgba_buf = io.BytesIO()
        Image.new("RGBA", (8, 8), (10, 20, 30, 128)).save(rgba_buf, format="PNG")
        rgba_png = rgba_buf.getvalue()
        req1 = FakeRequest(post_data={
            "name": "Y", "text": "",
            "image": FakeFileField("a.png", _real_png()),  # opaque -> jpg out
        })
        pid = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)["id"]
        first_path = self.mod._image_path_for(pid)
        self.assertEqual(first_path.suffix, ".jpg")
        req2 = FakeRequest(post_data={
            "id": pid, "name": "Y", "text": "",
            "image": FakeFileField("a.png", rgba_png),  # alpha -> png out
        })
        asyncio.run(self.mod.upsert_prompt(req2))
        self.assertFalse(first_path.exists())
        new_path = self.mod._image_path_for(pid)
        self.assertEqual(new_path.suffix, ".png")

    def test_upsert_rejects_invalid_id(self):
        req = FakeRequest(post_data={"id": "../etc", "name": "x", "text": ""})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)
        self.assertIn(b"invalid id", resp.body)

    def test_delete_route_removes_entry_and_image(self):
        req1 = FakeRequest(post_data={
            "name": "Z", "text": "",
            "image": FakeFileField("a.png", _real_png()),
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

    def test_save_node_requires_prompt_text(self):
        node = self.mod.PromptLibrarySave()
        with self.assertRaises(ValueError):
            node.save(name="Has Name", text="")
        with self.assertRaises(ValueError):
            node.save(name="Has Name", text="   \n  ")

    def test_save_node_with_thumbnail_writes_jpg(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("PIL/numpy not available")
        node = self.mod.PromptLibrarySave()
        img = self._fake_image()  # 3-channel RGB -> saved as JPEG by the resizer
        _, pid = node.save(name="ThumbTest", text="t", thumbnail=img)
        path = self.mod._image_path_for(pid)
        self.assertIsNotNone(path)
        self.assertEqual(path.suffix, ".jpg")
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
        added, updated, skipped, errors = self.mod._import_csv(text)
        self.assertEqual(added, 2)
        self.assertEqual(updated, 0)
        self.assertEqual(skipped, 0)
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
        # Then import a CSV that updates the same id — must opt into update mode
        # since the default is add_only (no overwrite).
        text = f"name,text,tags,id\nKnight,v2,character,{pid}\n"
        added, updated, skipped, errors = self.mod._import_csv(text, mode="update")
        self.assertEqual(added, 0)
        self.assertEqual(updated, 1)
        self.assertEqual(skipped, 0)
        items = self.mod._load()
        self.assertEqual(items[0]["text"], "v2")

    def test_import_csv_add_only_skips_existing(self):
        # Default add_only mode: existing entries are preserved verbatim.
        req = FakeRequest(post_data={"name": "Knight", "text": "v1", "tags": "original_tag"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        pid = body["id"]
        text = f"name,text,tags,id\nKnight,v2,clobbered_tag,{pid}\n"
        added, updated, skipped, errors = self.mod._import_csv(text)  # default add_only
        self.assertEqual(added, 0)
        self.assertEqual(updated, 0)
        self.assertEqual(skipped, 1)
        items = self.mod._load()
        # Original values preserved, NOT overwritten by CSV
        self.assertEqual(items[0]["text"], "v1")
        self.assertEqual(items[0]["tags"], ["original_tag"])

    def test_import_csv_skips_invalid_rows(self):
        text = "name,text,tags,id\n,nothing,,\nGood,t,,\nBad,t,,../etc\n"
        added, _, _, errors = self.mod._import_csv(text)
        self.assertEqual(added, 1)
        self.assertEqual(len(errors), 2)

    def test_import_csv_rejects_missing_name_column(self):
        text = "title,text\nfoo,bar\n"
        _, _, _, errors = self.mod._import_csv(text)
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

    # ---- wildcards -----------------------------------------------------

    def _seed_library(self, prompts):
        """Helper: write prompts as the library."""
        items = [{"id": p["id"], "name": p.get("name", p["id"]),
                  "text": p.get("text", ""), "tags": p.get("tags", [])} for p in prompts]
        self.mod._save(items)

    def test_wildcard_choice_deterministic_with_seed(self):
        import random
        rng = random.Random(0)
        result = self.mod._expand_wildcards("a {x|y|z} b", [], rng)
        self.assertIn(result, {"a x b", "a y b", "a z b"})
        # Same seed -> same result
        rng = random.Random(0)
        result2 = self.mod._expand_wildcards("a {x|y|z} b", [], rng)
        self.assertEqual(result, result2)

    def test_wildcard_weighted_choice_one_zero_picks_other(self):
        import random
        for seed in range(5):
            out = self.mod._expand_wildcards("{1::A|0::B}", [], random.Random(seed))
            self.assertEqual(out, "A")
            out = self.mod._expand_wildcards("{0::A|1::B}", [], random.Random(seed))
            self.assertEqual(out, "B")

    def test_wildcard_weighted_choice_distribution(self):
        import random
        rng = random.Random(42)
        counts = {"A": 0, "B": 0}
        for _ in range(2000):
            out = self.mod._expand_wildcards("{0.1::A|0.9::B}", [], rng)
            counts[out] += 1
        # 10/90 split: A should be roughly 200 / B roughly 1800. Wide tolerance
        # because RNG is real, not mocked.
        self.assertGreater(counts["B"], counts["A"] * 4)

    def test_wildcard_weighted_mixed_with_unweighted(self):
        # An unweighted alt defaults to weight 1.0; mix with explicit weights.
        import random
        rng = random.Random(0)
        out = self.mod._expand_wildcards("{a|0::b|0::c}", [], rng)
        self.assertEqual(out, "a")

    def test_wildcard_weighted_all_zero_falls_back_to_uniform(self):
        import random
        rng = random.Random(0)
        # All zero weights would be a div-by-zero; we fall back to uniform pick.
        out = self.mod._expand_wildcards("{0::a|0::b|0::c}", [], rng)
        self.assertIn(out, {"a", "b", "c"})

    def test_wildcard_weighted_malformed_prefix_treated_as_literal(self):
        # "abc::x" is not a numeric weight — keep the whole thing as a literal.
        import random
        out = self.mod._expand_wildcards("{abc::x|1::y}", [], random.Random(0))
        # Either alt may win on different seeds, but both are kept; verify the
        # malformed one is preserved verbatim when it does win.
        results = set()
        for seed in range(20):
            results.add(self.mod._expand_wildcards("{abc::x|abc::y}", [], random.Random(seed)))
        self.assertTrue(results.issubset({"abc::x", "abc::y"}))

    def test_wildcard_choice_no_alternation_strips_braces(self):
        import random
        out = self.mod._expand_wildcards("hello {world}", [], random.Random(0))
        self.assertEqual(out, "hello world")

    def test_wildcard_named_ref_by_id(self):
        import random
        self._seed_library([{"id": "knight", "text": "armored knight"}])
        out = self.mod._expand_wildcards("a __knight__ here", self.mod._load(), random.Random(0))
        self.assertEqual(out, "a armored knight here")

    def test_wildcard_named_ref_by_tag_random(self):
        import random
        self._seed_library([
            {"id": "elf", "text": "elf", "tags": ["character"]},
            {"id": "wizard", "text": "wizard", "tags": ["character"]},
        ])
        rng = random.Random(0)
        out = self.mod._expand_wildcards("a __character__", self.mod._load(), rng)
        self.assertIn(out, {"a elf", "a wizard"})

    def test_wildcard_unknown_ref_left_literal(self):
        import random
        out = self.mod._expand_wildcards("a __missing__ b", [], random.Random(0))
        self.assertEqual(out, "a __missing__ b")

    def test_wildcard_recursion_expands_nested(self):
        import random
        self._seed_library([
            {"id": "wizard", "text": "{old|young} wizard"},
            {"id": "knight", "text": "knight"},
        ])
        rng = random.Random(0)
        out = self.mod._expand_wildcards("__wizard__", self.mod._load(), rng)
        self.assertIn(out, {"old wizard", "young wizard"})

    def test_wildcard_cycle_protection(self):
        # a -> b -> a -> b -> ... should bottom out at the depth cap.
        import random
        self._seed_library([
            {"id": "a", "text": "__b__"},
            {"id": "b", "text": "__a__"},
        ])
        out = self.mod._expand_wildcards("__a__", self.mod._load(), random.Random(0))
        # Should terminate without infinite loop; final state likely contains __a__ or __b__.
        self.assertIn(out, {"__a__", "__b__"})

    def test_wildcard_choice_with_named_ref_inside(self):
        import random
        self._seed_library([{"id": "wizard", "text": "wizard"}])
        rng = random.Random(0)
        out = self.mod._expand_wildcards("{__wizard__|knight}", self.mod._load(), rng)
        self.assertIn(out, {"wizard", "knight"})

    # ---- Rating + notes ------------------------------------------------

    def test_upsert_stores_rating_and_notes(self):
        req = FakeRequest(post_data={
            "name": "Test", "text": "hello", "tags": "",
            "rating": "4", "notes": "good for night scenes",
        })
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        self.assertEqual(body["rating"], 4)
        self.assertEqual(body["notes"], "good for night scenes")
        with self.mod._lock:
            stored = self.mod._load()
        self.assertEqual(stored[0]["rating"], 4)
        self.assertEqual(stored[0]["notes"], "good for night scenes")

    def test_upsert_rating_clamped_to_0_5(self):
        for raw, want in (("0", 0), ("5", 5), ("9", 5), ("-3", 0)):
            req = FakeRequest(post_data={"name": f"P-{raw}", "text": "x", "rating": raw})
            body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
            self.assertEqual(body["rating"], want, f"input {raw!r} should clamp to {want}")

    def test_upsert_invalid_rating_returns_400(self):
        req = FakeRequest(post_data={"name": "P", "text": "x", "rating": "abc"})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)

    def test_upsert_thumbnail_failure_rolls_back_new_entry(self):
        # A bogus image (decodable as bytes but not a valid image) should
        # fail thumbnail processing AND not leave a half-written entry.
        req = FakeRequest(post_data={
            "name": "BadImg", "text": "x",
            "image": FakeFileField("x.png", b"not actually an image"),
        })
        resp = asyncio.run(self.mod.upsert_prompt(req))
        self.assertEqual(resp.status, 400)
        with self.mod._lock:
            self.assertEqual(self.mod._load(), [],
                             "failed thumbnail upload should not persist a new entry")

    def test_upsert_thumbnail_failure_rolls_back_existing_entry(self):
        # Create an entry, then try to update it with a corrupt image.
        first = FakeRequest(post_data={"name": "Good", "text": "v1", "tags": "fantasy"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(first)).body)
        pid = body["id"]
        bad = FakeRequest(post_data={
            "id": pid, "name": "ChangedName", "text": "v2", "tags": "scifi",
            "image": FakeFileField("x.png", b"not actually an image"),
        })
        resp = asyncio.run(self.mod.upsert_prompt(bad))
        self.assertEqual(resp.status, 400)
        with self.mod._lock:
            stored = self.mod._load()
        self.assertEqual(stored[0]["name"], "Good", "name should be rolled back")
        self.assertEqual(stored[0]["text"], "v1", "text should be rolled back")
        self.assertEqual(stored[0]["tags"], ["fantasy"], "tags should be rolled back")

    def test_export_import_round_trips_rating_and_notes(self):
        req = FakeRequest(post_data={
            "name": "RatedEntry", "text": "x", "rating": "3", "notes": "hi",
        })
        asyncio.run(self.mod.upsert_prompt(req))
        with self.mod._lock:
            items = self.mod._load()
        zip_bytes = self.mod._build_export_zip(items, "test")
        self.mod._save([])
        # add_only is fine here — the library is empty after _save([])
        self.mod._import_zip(zip_bytes)
        with self.mod._lock:
            restored = self.mod._load()
        self.assertEqual(restored[0]["rating"], 3)
        self.assertEqual(restored[0]["notes"], "hi")

    # ---- PromptLibraryMulti node ---------------------------------------

    def test_multi_node_three_outputs(self):
        self.assertIn("PromptLibraryMulti", self.mod.NODE_CLASS_MAPPINGS)
        cls = self.mod.PromptLibraryMulti
        self.assertEqual(len(cls.RETURN_TYPES), 3)
        self.assertEqual(cls.RETURN_NAMES, ("prompt_1", "prompt_2", "prompt_3"))

    def test_multi_node_independent_panels(self):
        self._seed_library([
            {"id": "frieren", "text": "frieren", "tags": ["character"]},
            {"id": "anime",   "text": "anime style",  "tags": ["style"]},
            {"id": "hoodie",  "text": "hoodie",       "tags": ["clothing"]},
        ])
        node = self.mod.PromptLibraryMulti()
        out = node.load_prompts(
            prompt_id_1="frieren",
            prompt_id_2="anime",
            prompt_id_3="hoodie",
            separator_1=", ", separator_2=", ", separator_3=", ",
            label_1="Character", label_2="Style", label_3="Clothing",
        )
        self.assertEqual(out, ("frieren", "anime style", "hoodie"))

    def test_multi_node_joins_multiple_ids(self):
        self._seed_library([
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "bravo"},
        ])
        node = self.mod.PromptLibraryMulti()
        out = node.load_prompts(
            prompt_id_1="a,b",
            prompt_id_2="",
            prompt_id_3="",
            separator_1=" | ",
        )
        self.assertEqual(out[0], "alpha | bravo")
        self.assertEqual(out[1], "")
        self.assertEqual(out[2], "")

    def test_multi_node_skips_missing_ids(self):
        self._seed_library([{"id": "a", "text": "alpha"}])
        node = self.mod.PromptLibraryMulti()
        out = node.load_prompts(prompt_id_1="a,nope", prompt_id_2="", prompt_id_3="")
        self.assertEqual(out[0], "alpha")

    # ---- Comic-strip nodes ---------------------------------------------

    def test_scene_node_joins_non_empty_fields(self):
        node = self.mod.PromptLibraryScene()
        out, = node.build(
            time_of_day="dusk", weather="", lighting="warm orange glow",
            camera_angle="low angle", mood="", framing="rule of thirds",
            extra="50mm lens", separator=", ",
        )
        self.assertEqual(out, "dusk, warm orange glow, low angle, rule of thirds, 50mm lens")

    def test_scene_node_all_blank_returns_empty(self):
        node = self.mod.PromptLibraryScene()
        out, = node.build("", "", "", "", "", "", "", separator=", ")
        self.assertEqual(out, "")

    def test_scene_node_strips_whitespace(self):
        node = self.mod.PromptLibraryScene()
        out, = node.build("  morning  ", "", "  ", "", "", "", "", separator=", ")
        self.assertEqual(out, "morning")

    def test_background_node_preset_only(self):
        # Pick the first non-divider, non-(none) preset so this test stays
        # robust against future expansions of the preset list.
        node = self.mod.PromptLibraryBackground()
        first_real = next(p for p in self.mod._BG_PRESETS
                          if p != self.mod._BG_NONE
                          and not p.startswith(self.mod._BG_DIVIDER_CHAR))
        out, = node.build(preset=first_real, custom="", separator=", ")
        self.assertEqual(out, first_real)

    def test_background_node_custom_only(self):
        node = self.mod.PromptLibraryBackground()
        out, = node.build(preset=self.mod._BG_NONE,
                           custom="abandoned shrine in the woods", separator=", ")
        self.assertEqual(out, "abandoned shrine in the woods")

    def test_background_node_combines_preset_and_custom(self):
        node = self.mod.PromptLibraryBackground()
        first_real = next(p for p in self.mod._BG_PRESETS
                          if p != self.mod._BG_NONE
                          and not p.startswith(self.mod._BG_DIVIDER_CHAR))
        out, = node.build(preset=first_real,
                           custom="rain pattering on the window", separator=", ")
        self.assertEqual(out, first_real + ", rain pattering on the window")

    def test_background_node_both_empty_returns_empty(self):
        node = self.mod.PromptLibraryBackground()
        out, = node.build(preset=self.mod._BG_NONE, custom="", separator=", ")
        self.assertEqual(out, "")

    def test_background_node_divider_row_treated_as_none(self):
        # Visual divider rows like "───── home / interior ─────" are visible
        # in the dropdown but must not pollute the prompt if clicked.
        node = self.mod.PromptLibraryBackground()
        divider = self.mod._bg_divider("home / interior")
        out, = node.build(preset=divider, custom="rainy night", separator=", ")
        self.assertEqual(out, "rainy night")
        out2, = node.build(preset=divider, custom="", separator=", ")
        self.assertEqual(out2, "")

    def test_background_presets_include_hallways_and_rooms(self):
        # User-requested categories are present.
        presets = self.mod._BG_PRESETS
        self.assertTrue(any(p.startswith("hallway:") for p in presets),
                        "expected hallway: entries in the preset list")
        self.assertTrue(any("master bedroom" in p for p in presets),
                        "expected master bedroom in the preset list")
        self.assertTrue(any("dining room" in p for p in presets),
                        "expected dining room in the preset list")

    def test_scene_node_treats_none_sentinel_as_empty(self):
        node = self.mod.PromptLibraryScene()
        out, = node.build(
            time_of_day=self.mod._SCENE_NONE, weather="light rain",
            lighting=self.mod._SCENE_NONE, camera_angle="low angle",
            mood=self.mod._SCENE_NONE, framing=self.mod._SCENE_NONE,
            extra="", separator=", ",
        )
        self.assertEqual(out, "light rain, low angle")

    def test_comic_frame_combines_anchors_and_action(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, seed, count = node.assemble(
            frames_json='["kicks the door open", "phone rings", "answers, surprised"]',
            frame_index=2, separator=", ",
            character="frieren, white hair",
            scene="dusk, warm light",
            background="abandoned warehouse",
            base_seed=1000,
        )
        self.assertEqual(prompt, "frieren, white hair, dusk, warm light, abandoned warehouse, phone rings")
        self.assertEqual(action, "phone rings")
        self.assertEqual(seed, 1001)
        self.assertEqual(count, 3)

    def test_comic_frame_clamps_index_above_range(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, seed, count = node.assemble(
            frames_json='["a", "b"]', frame_index=99,
            character="x",
        )
        self.assertEqual(action, "b")  # clamped to last frame
        self.assertEqual(prompt, "x, b")

    def test_comic_frame_clamps_index_below_range(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, _seed, _count = node.assemble(
            frames_json='["a", "b"]', frame_index=0,
        )
        self.assertEqual(action, "a")

    def test_comic_frame_no_frames_emits_anchors_only(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, _seed, count = node.assemble(
            frames_json='[]', frame_index=1,
            character="hero", scene="night", background="rooftop",
        )
        self.assertEqual(prompt, "hero, night, rooftop")
        self.assertEqual(action, "")
        self.assertEqual(count, 0)

    def test_comic_frame_invalid_frames_json_falls_back(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, _, count = node.assemble(
            frames_json='not valid json', frame_index=1, character="x",
        )
        self.assertEqual(prompt, "x")
        self.assertEqual(action, "")
        self.assertEqual(count, 0)

    def test_comic_frame_seed_offset_per_frame(self):
        node = self.mod.PromptLibraryComicFrame()
        seeds = []
        for i in range(1, 4):
            _, _, seed, _ = node.assemble(
                frames_json='["a","b","c"]', frame_index=i, base_seed=12345,
            )
            seeds.append(seed)
        self.assertEqual(seeds, [12345, 12346, 12347])

    # ---- PromptLibraryRandom node --------------------------------------

    def test_random_picks_by_tag(self):
        self._seed_library([
            {"id": "elf", "text": "elf", "tags": ["character", "fantasy"]},
            {"id": "wizard", "text": "wizard", "tags": ["character", "fantasy"]},
            {"id": "noir", "text": "noir city", "tags": ["style"]},
        ])
        node = self.mod.PromptLibraryRandom()
        text, pid = node.pick(tag_filter="character", seed=0)
        self.assertIn(pid, {"elf", "wizard"})
        self.assertEqual(text, pid)

    def test_random_and_filter_multi_tag(self):
        self._seed_library([
            {"id": "elf", "text": "elf", "tags": ["character", "fantasy"]},
            {"id": "wizard", "text": "wizard", "tags": ["character", "fantasy"]},
            {"id": "robot", "text": "robot", "tags": ["character", "scifi"]},
        ])
        node = self.mod.PromptLibraryRandom()
        text, pid = node.pick(tag_filter="character, fantasy", seed=0)
        self.assertIn(pid, {"elf", "wizard"})

    def test_random_no_match_returns_empty(self):
        self._seed_library([{"id": "x", "text": "x", "tags": ["a"]}])
        node = self.mod.PromptLibraryRandom()
        out = node.pick(tag_filter="missing", seed=0)
        self.assertEqual(out, ("", ""))

    def test_random_empty_filter_picks_anything(self):
        self._seed_library([
            {"id": "a", "text": "a"},
            {"id": "b", "text": "b"},
        ])
        node = self.mod.PromptLibraryRandom()
        _, pid = node.pick(tag_filter="", seed=0)
        self.assertIn(pid, {"a", "b"})

    def test_random_seed_determinism(self):
        self._seed_library([{"id": f"p{i}", "text": f"t{i}", "tags": ["x"]} for i in range(10)])
        node = self.mod.PromptLibraryRandom()
        a = node.pick(tag_filter="x", seed=42)
        b = node.pick(tag_filter="x", seed=42)
        self.assertEqual(a, b)

    def test_random_expands_wildcards_in_picked(self):
        self._seed_library([
            {"id": "knight", "text": "{red|blue} knight", "tags": ["character"]},
        ])
        node = self.mod.PromptLibraryRandom()
        text, pid = node.pick(tag_filter="character", seed=0, expand_wildcards=True)
        self.assertIn(text, {"red knight", "blue knight"})
        self.assertEqual(pid, "knight")

    # ---- PromptLibraryWildcard node ------------------------------------

    def test_wildcard_node_expands(self):
        self._seed_library([{"id": "elf", "text": "tall elf"}])
        node = self.mod.PromptLibraryWildcard()
        out = node.expand(text="hi __elf__ {a|b}", seed=0)
        self.assertTrue(out[0].startswith("hi tall elf "))
        self.assertIn(out[0][-1], {"a", "b"})

    def test_wildcard_node_toggle_choices_off(self):
        node = self.mod.PromptLibraryWildcard()
        out = node.expand(text="{a|b}", seed=0, expand_choices=False, expand_named_refs=True)
        self.assertEqual(out[0], "{a|b}")

    def test_wildcard_node_toggle_named_off(self):
        self._seed_library([{"id": "elf", "text": "tall elf"}])
        node = self.mod.PromptLibraryWildcard()
        out = node.expand(text="__elf__ {x|y}", seed=0, expand_choices=True, expand_named_refs=False)
        self.assertIn(out[0], {"__elf__ x", "__elf__ y"})

    def test_wildcard_node_both_off_passthrough(self):
        node = self.mod.PromptLibraryWildcard()
        out = node.expand(text="__a__ {x|y}", seed=0, expand_choices=False, expand_named_refs=False)
        self.assertEqual(out[0], "__a__ {x|y}")

    # ---- ZIP export / import -------------------------------------------

    def test_export_zip_roundtrip(self):
        # Create entries (one with image), export, wipe, re-import.
        png_bytes = _real_png()
        req = FakeRequest(post_data={
            "name": "Knight", "text": "armored knight", "tags": "character, fantasy",
            "image": FakeFileField("knight.png", png_bytes),
        })
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        knight_id = body["id"]
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={
            "name": "Wizard", "text": "old wizard", "tags": "character",
        })))

        # Export both
        with self.mod._lock:
            items = self.mod._load()
        zip_bytes = self.mod._build_export_zip(items, "test-1.0")
        self.assertGreater(len(zip_bytes), 100)

        # Wipe and re-import
        self.mod._save([])
        self.mod._delete_image_files(knight_id)
        added, updated, skipped, errors = self.mod._import_zip(zip_bytes)
        self.assertEqual(added, 2)
        self.assertEqual(updated, 0)
        self.assertEqual(skipped, 0)
        self.assertEqual(errors, [])
        items = self.mod._load()
        self.assertEqual(len(items), 2)
        # Knight image preserved
        self.assertIsNotNone(self.mod._image_path_for(knight_id))

    def test_export_route_returns_count_header(self):
        # Two entries, no filter — header should report total count.
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"name": "A", "text": "a"})))
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"name": "B", "text": "b"})))
        req = FakeRequest(json_data={"ids": []})
        # FakeRequest needs body_exists so the route reads json
        req.body_exists = True
        resp = asyncio.run(self.mod.export_zip(req))
        self.assertEqual(resp.headers["X-GrimmRibbity-Count"], "2")
        self.assertEqual(resp.headers["Content-Type"], "application/zip")

    def test_export_route_filters_by_ids(self):
        a_id = json.loads(asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"name": "A", "text": "a"}))).body)["id"]
        json.loads(asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={"name": "B", "text": "b"}))).body)
        req = FakeRequest(json_data={"ids": [a_id]})
        req.body_exists = True
        resp = asyncio.run(self.mod.export_zip(req))
        self.assertEqual(resp.headers["X-GrimmRibbity-Count"], "1")

    def test_import_zip_updates_existing(self):
        # First entry
        req1 = FakeRequest(post_data={"name": "X", "text": "v1"})
        pid = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)["id"]
        # Build a zip describing the same id but different text
        items = [{"id": pid, "name": "X", "text": "v2 from zip", "tags": []}]
        zip_bytes = self.mod._build_export_zip(items, "test")
        # Must opt into mode=update — default is add_only.
        added, updated, skipped, errors = self.mod._import_zip(zip_bytes, mode="update")
        self.assertEqual(added, 0)
        self.assertEqual(updated, 1)
        self.assertEqual(skipped, 0)
        entry = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertEqual(entry["text"], "v2 from zip")
        self.assertEqual(len(entry["history"]), 1)
        self.assertEqual(entry["history"][0]["text"], "v1")

    def test_import_zip_add_only_skips_existing(self):
        # Default add_only mode: existing entries are preserved verbatim.
        req = FakeRequest(post_data={"name": "X", "text": "original", "tags": "keep"})
        pid = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)["id"]
        items = [{"id": pid, "name": "X", "text": "clobbered", "tags": ["wrong"]}]
        zip_bytes = self.mod._build_export_zip(items, "test")
        added, updated, skipped, errors = self.mod._import_zip(zip_bytes)  # default add_only
        self.assertEqual(added, 0)
        self.assertEqual(updated, 0)
        self.assertEqual(skipped, 1)
        entry = next(i for i in self.mod._load() if i["id"] == pid)
        self.assertEqual(entry["text"], "original")
        self.assertEqual(entry["tags"], ["keep"])

    def test_import_zip_rejects_bad_zip(self):
        added, updated, skipped, errors = self.mod._import_zip(b"not a zip")
        self.assertEqual((added, updated, skipped), (0, 0, 0))
        self.assertTrue(errors)

    def test_import_zip_rejects_missing_manifest(self):
        buf = io.BytesIO()
        with __import__("zipfile").ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "hello")
        added, updated, skipped, errors = self.mod._import_zip(buf.getvalue())
        self.assertEqual((added, updated, skipped), (0, 0, 0))
        self.assertTrue(any("prompts.json" in e for e in errors))

    def test_import_zip_skips_invalid_id(self):
        items = [{"id": "../bad", "name": "X", "text": "t"}]
        zip_bytes = self.mod._build_export_zip(items, "test")
        added, updated, skipped, errors = self.mod._import_zip(zip_bytes)
        self.assertEqual(added, 0)
        self.assertTrue(any("invalid id" in e for e in errors))

    def test_import_zip_skips_empty_name(self):
        items = [{"id": "abc", "name": "", "text": "t"}]
        zip_bytes = self.mod._build_export_zip(items, "test")
        added, updated, skipped, errors = self.mod._import_zip(zip_bytes)
        self.assertEqual(added, 0)
        self.assertTrue(any("empty name" in e for e in errors))

    # ---- snapshot / undo ----------------------------------------------

    def test_snapshot_and_restore(self):
        # Seed two entries, snapshot, mutate, restore — should bring back
        # the snapshotted state.
        self.mod._save([{"id": "a", "name": "A", "text": "v1"},
                          {"id": "b", "name": "B", "text": "v1"}])
        snap_name = self.mod._snapshot_prompts("test_pre")
        self.assertTrue(snap_name)
        # Mutate
        self.mod._save([{"id": "a", "name": "A", "text": "MUTATED"}])
        result = self.mod._restore_snapshot(snap_name)
        self.assertTrue(result["ok"])
        self.assertEqual(result["entries"], 2)
        items = self.mod._load()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["text"], "v1")

    def test_restore_snapshot_rejects_path_traversal(self):
        result = self.mod._restore_snapshot("../prompts.json")
        self.assertFalse(result["ok"])
        self.assertIn("invalid", result["error"].lower())

    def test_restore_snapshot_handles_missing(self):
        result = self.mod._restore_snapshot("nonexistent.json")
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    # ---- auto-backup --------------------------------------------------

    def test_autobackup_skips_first_run(self):
        # First run: no marker, no backup created.
        self.mod._save([{"id": "x", "name": "X", "text": "t"}])
        self.mod._autobackup_on_version_change()
        backups = [p for p in self.mod.ROOT.iterdir() if p.is_dir() and p.name.startswith("data-backup-")]
        self.assertEqual(backups, [])

    def test_autobackup_on_version_change(self):
        # Seed a marker with an older version.
        (self.mod.DATA_DIR / ".last_version").write_text("0.0.1")
        self.mod._save([{"id": "x", "name": "X", "text": "t"}])
        self.mod._autobackup_on_version_change()
        backups = sorted(p.name for p in self.mod.ROOT.iterdir()
                         if p.is_dir() and p.name.startswith("data-backup-0.0.1-"))
        self.assertEqual(len(backups), 1)
        # Backup contains the prompts.json from before the upgrade
        backed_up = self.mod.ROOT / backups[0] / "prompts.json"
        self.assertTrue(backed_up.exists())
        self.assertIn('"id": "x"', backed_up.read_text())

    def test_autobackup_no_backup_when_version_unchanged(self):
        (self.mod.DATA_DIR / ".last_version").write_text(self.mod.__version__)
        self.mod._save([{"id": "x", "name": "X", "text": "t"}])
        self.mod._autobackup_on_version_change()
        backups = [p for p in self.mod.ROOT.iterdir() if p.is_dir() and p.name.startswith("data-backup-")]
        self.assertEqual(backups, [])

    # ---- bulk delete / duplicate / reorder -----------------------------

    def test_bulk_delete_removes_listed_entries(self):
        self.mod._save([
            {"id": "a", "name": "A", "text": "1"},
            {"id": "b", "name": "B", "text": "2"},
            {"id": "c", "name": "C", "text": "3"},
        ])
        req = FakeRequest(json_data={"ids": ["a", "c"]})
        resp = asyncio.run(self.mod.bulk_delete(req))
        body = json.loads(resp.body)
        self.assertEqual(body["deleted"], 2)
        remaining = [i["id"] for i in self.mod._load()]
        self.assertEqual(remaining, ["b"])

    def test_bulk_delete_skips_invalid_ids(self):
        self.mod._save([{"id": "a", "name": "A", "text": "1"}])
        req = FakeRequest(json_data={"ids": ["a", "../bad", ""]})
        resp = asyncio.run(self.mod.bulk_delete(req))
        body = json.loads(resp.body)
        self.assertEqual(body["deleted"], 1)

    def test_bulk_delete_rejects_no_valid_ids(self):
        req = FakeRequest(json_data={"ids": ["../bad", ""]})
        resp = asyncio.run(self.mod.bulk_delete(req))
        self.assertEqual(resp.status, 400)

    def test_duplicate_creates_copy_with_new_id(self):
        png = _real_png()
        req1 = FakeRequest(post_data={
            "name": "Original", "text": "body", "tags": "tag1",
            "image": FakeFileField("o.png", png),
        })
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        original_id = body["id"]

        dup_req = FakeRequest(json_data={"id": original_id})
        resp = asyncio.run(self.mod.duplicate_prompt(dup_req))
        dup = json.loads(resp.body)
        self.assertNotEqual(dup["id"], original_id)
        self.assertEqual(dup["name"], "Original (copy)")

        items = self.mod._load()
        self.assertEqual(len(items), 2)
        new_entry = next(i for i in items if i["id"] == dup["id"])
        self.assertEqual(new_entry["text"], "body")
        self.assertEqual(new_entry["tags"], ["tag1"])
        # Image was copied
        self.assertIsNotNone(self.mod._image_path_for(dup["id"]))

    def test_duplicate_with_custom_name(self):
        self.mod._save([{"id": "src", "name": "Src", "text": "t"}])
        req = FakeRequest(json_data={"id": "src", "name": "Custom Copy"})
        resp = asyncio.run(self.mod.duplicate_prompt(req))
        dup = json.loads(resp.body)
        self.assertEqual(dup["name"], "Custom Copy")
        self.assertEqual(dup["id"], "custom_copy")

    def test_duplicate_404_for_unknown(self):
        req = FakeRequest(json_data={"id": "nope"})
        resp = asyncio.run(self.mod.duplicate_prompt(req))
        self.assertEqual(resp.status, 404)

    def test_reorder_applies_order_field(self):
        self.mod._save([
            {"id": "a", "name": "A", "text": "1"},
            {"id": "b", "name": "B", "text": "2"},
            {"id": "c", "name": "C", "text": "3"},
        ])
        req = FakeRequest(json_data={"ids": ["c", "a", "b"]})
        resp = asyncio.run(self.mod.reorder_prompts(req))
        body = json.loads(resp.body)
        self.assertEqual(body["count"], 3)
        items = self.mod._load()
        self.assertEqual([i["id"] for i in items], ["c", "a", "b"])
        self.assertEqual(items[0]["order"], 0)
        self.assertEqual(items[2]["order"], 2)

    def test_reorder_handles_partial_list(self):
        self.mod._save([
            {"id": "a", "name": "A", "text": "1"},
            {"id": "b", "name": "B", "text": "2"},
            {"id": "c", "name": "C", "text": "3"},
        ])
        # Only specify b first; a and c keep their relative position after.
        req = FakeRequest(json_data={"ids": ["b"]})
        asyncio.run(self.mod.reorder_prompts(req))
        items = self.mod._load()
        self.assertEqual(items[0]["id"], "b")
        # a and c still in order, after b
        rest = [i["id"] for i in items[1:]]
        self.assertEqual(rest, ["a", "c"])

    def test_reorder_rejects_invalid_id(self):
        self.mod._save([{"id": "a", "name": "A", "text": "1"}])
        req = FakeRequest(json_data={"ids": ["a", "../bad"]})
        resp = asyncio.run(self.mod.reorder_prompts(req))
        self.assertEqual(resp.status, 400)

    # ---- watcher (smoke) -----------------------------------------------

    def test_watcher_starts_when_invoked(self):
        # Auto-start is skipped under unittest; verify _start_watcher runs cleanly.
        import threading
        before = sum(1 for t in threading.enumerate() if t.name == "prompt-library-watcher")
        self.mod._start_watcher()
        after = sum(1 for t in threading.enumerate() if t.name == "prompt-library-watcher")
        self.assertEqual(after, before + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
