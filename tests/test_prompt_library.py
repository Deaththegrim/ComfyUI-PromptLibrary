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
        self.assertEqual(self.mod.WEB_DIRECTORY, "./web")


if __name__ == "__main__":
    unittest.main(verbosity=2)
