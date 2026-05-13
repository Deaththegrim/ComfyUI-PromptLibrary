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
    mod.SNAPSHOT_DIR = mod.DATA_DIR / "snapshots"
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
        # Mirror aiohttp's body_exists attribute — used by the route's
        # _json_payload helper to short-circuit empty POSTs.
        self.body_exists = bool(self._post_data) or bool(self._json_data)

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
        self.assertFalse(self.mod.STORE_PATH.exists())
        broken = list(self.mod.DATA_DIR.glob("prompts.json.broken-*"))
        self.assertEqual(len(broken), 1)

    def test_load_restores_from_snapshot_when_corrupted(self):
        good = [{"id": "k1", "name": "n", "text": "hello"}]
        self.mod._save(good)
        self.mod._snapshot_prompts("manual_test")
        self.mod.STORE_PATH.write_text("not json {{{")
        self.assertEqual(self.mod._load(), good)
        broken = list(self.mod.DATA_DIR.glob("prompts.json.broken-*"))
        self.assertEqual(len(broken), 1)

    def test_load_handles_non_list_json(self):
        self.mod.STORE_PATH.write_text('{"foo": "bar"}')
        self.assertEqual(self.mod._load(), [])

    def test_load_heals_export_manifest_into_storage_list(self):
        manifest = {
            "format": "grimm-ribbity-prompt-library",
            "format_version": 1,
            "exported_with": "0.27.3",
            "prompts": [{"id": "k1", "name": "n", "text": "hello"}],
        }
        self.mod.STORE_PATH.write_text(json.dumps(manifest))
        self.assertEqual(self.mod._load(), manifest["prompts"])
        with self.mod.STORE_PATH.open() as f:
            self.assertIsInstance(json.load(f), list)
        snaps = list(self.mod.SNAPSHOT_DIR.glob("*pre_heal_manifest*.json"))
        self.assertEqual(len(snaps), 1)

    def test_load_falls_through_when_snapshot_also_corrupt(self):
        # Both prompts.json AND the only snapshot are unparseable — must not
        # crash, must quarantine the live file, must return empty list.
        self.mod._save([{"id": "k1", "name": "n", "text": "hello"}])
        self.mod._snapshot_prompts("manual_test")
        snap = next(iter(self.mod.SNAPSHOT_DIR.glob("*.json")))
        snap.write_text("also broken {{{")
        self.mod.STORE_PATH.write_text("not json {{{")
        self.assertEqual(self.mod._load(), [])
        broken = list(self.mod.DATA_DIR.glob("prompts.json.broken-*"))
        self.assertEqual(len(broken), 1)

    def test_load_picks_newest_snapshot_among_many(self):
        # Two snapshots — older has stale data, newer has the right entries.
        # The recovery path should pick the newer one.
        old = [{"id": "old", "name": "n", "text": "old"}]
        new = [{"id": "new1", "name": "n", "text": "new"},
               {"id": "new2", "name": "n", "text": "new2"}]
        self.mod.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        (self.mod.SNAPSHOT_DIR / "20260101-000000-a.json").write_text(json.dumps(old))
        (self.mod.SNAPSHOT_DIR / "20260601-120000-b.json").write_text(json.dumps(new))
        self.mod.STORE_PATH.write_text("not json {{{")
        self.assertEqual(self.mod._load(), new)

    def test_safe_id_accepts_valid(self):
        for ok in ["abc", "a_b-c", "ABC123", "x" * 64]:
            self.assertEqual(self.mod._safe_id(ok), ok)

    def test_safe_id_rejects_invalid(self):
        for bad in ["", "..", "a/b", "with space", "x" * 65, "café", None and ""]:
            self.assertIsNone(self.mod._safe_id(bad or ""))

    # ---- node logic -----------------------------------------------------

    def _strip_lora_outputs(self, result):
        """The Library node now returns (prompt, negative, model, clip).
        Tests that pre-date the LoRA-aware sockets only care about the
        first two; this helper trims the rest so they read cleanly."""
        return result[:2]

    def test_load_prompt_returns_text_for_known_id(self):
        self.mod._save([{"id": "k1", "name": "n", "text": "hello"}])
        node = self.mod.PromptLibrary()
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("k1")),
                          ("hello", ""))

    def test_load_prompt_empty_for_missing_id(self):
        node = self.mod.PromptLibrary()
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("missing")),
                          ("", ""))

    def test_load_prompt_empty_for_blank_id(self):
        node = self.mod.PromptLibrary()
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("")),
                          ("", ""))

    def test_load_prompt_joins_multiple_ids_with_separator(self):
        self.mod._save([
            {"id": "a", "name": "A", "text": "alpha"},
            {"id": "b", "name": "B", "text": "beta"},
            {"id": "c", "name": "C", "text": "gamma"},
        ])
        node = self.mod.PromptLibrary()
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("a,b,c")),
                          ("alpha, beta, gamma", ""))
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("a, b , c")),
                          ("alpha, beta, gamma", ""))
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("a,c", separator=" | ")),
                          ("alpha | gamma", ""))

    def test_load_prompt_skips_missing_in_multi_id(self):
        self.mod._save([{"id": "a", "name": "A", "text": "alpha"}])
        node = self.mod.PromptLibrary()
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("a,missing,a")),
                          ("alpha, alpha", ""))

    def test_load_prompt_returns_negative_when_present(self):
        self.mod._save([{"id": "k1", "name": "n", "text": "hello",
                         "negative": "lowres, bad_anatomy"}])
        node = self.mod.PromptLibrary()
        self.assertEqual(self._strip_lora_outputs(node.load_prompt("k1")),
                          ("hello", "lowres, bad_anatomy"))

    def test_load_prompt_joins_negatives_skipping_empties(self):
        # Three entries: only the middle one has a negative — joined output
        # should contain just that one (no leading/trailing separator junk).
        self.mod._save([
            {"id": "a", "name": "A", "text": "alpha"},
            {"id": "b", "name": "B", "text": "beta", "negative": "blurry"},
            {"id": "c", "name": "C", "text": "gamma", "negative": "watermark"},
        ])
        node = self.mod.PromptLibrary()
        out = node.load_prompt("a,b,c")
        self.assertEqual(out[0], "alpha, beta, gamma")
        self.assertEqual(out[1], "blurry, watermark")

    def test_load_prompt_unwired_model_clip_pass_none(self):
        """When MODEL/CLIP aren't wired, the corresponding outputs are None
        and no LoRA work is attempted (no torch needed)."""
        self.mod._save([{"id": "k1", "name": "n", "text": "hello",
                         "loras": [{"name": "x.safetensors", "strength_model": 1.0}]}])
        node = self.mod.PromptLibrary()
        out = node.load_prompt("k1")
        self.assertEqual(out[:2], ("hello", ""))
        self.assertIsNone(out[2])
        self.assertIsNone(out[3])

    def test_is_changed_reflects_text(self):
        # IS_CHANGED hash includes the entry's text + negative so changes on
        # either side invalidate ComfyUI's cached output. The exact format
        # is opaque — we just need different values to produce different keys.
        self.mod._save([{"id": "k", "name": "n", "text": "v1"}])
        h_v1 = self.mod.PromptLibrary.IS_CHANGED("k")
        self.assertIn("v1", h_v1)
        self.mod._save([{"id": "k", "name": "n", "text": "v2"}])
        h_v2 = self.mod.PromptLibrary.IS_CHANGED("k")
        self.assertNotEqual(h_v1, h_v2)
        self.mod._save([{"id": "k", "name": "n", "text": "v2", "negative": "blurry"}])
        h_v2_neg = self.mod.PromptLibrary.IS_CHANGED("k")
        self.assertNotEqual(h_v2, h_v2_neg)
        self.assertIn("blurry", h_v2_neg)

    def test_is_changed_includes_loras_only_when_lora_aware(self):
        """Editing an entry's LoRA stack shouldn't invalidate the cache for
        STRING-only consumers (no MODEL/CLIP wired) — that would force a
        re-run on every LoRA tweak even though the output text is identical."""
        self.mod._save([{"id": "k", "name": "n", "text": "v"}])
        h_no_lora = self.mod.PromptLibrary.IS_CHANGED("k")
        self.mod._save([{"id": "k", "name": "n", "text": "v",
                          "loras": [{"name": "x.safetensors", "strength_model": 1.0,
                                     "strength_clip": 1.0, "triggers": "", "enabled": True}]}])
        h_with_lora = self.mod.PromptLibrary.IS_CHANGED("k")
        self.assertEqual(h_no_lora, h_with_lora)
        # But with model+clip wired (sentinel objects work — IS_CHANGED only
        # checks for not-None), the LoRA signature gets folded in.
        h_lora_aware = self.mod.PromptLibrary.IS_CHANGED(
            "k", model=object(), clip=object())
        self.assertNotEqual(h_no_lora, h_lora_aware)

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

    def test_list_route_exposes_negative_field(self):
        # The frontend modal needs /list to surface negative so the textarea
        # repopulates on edit. Default empty for entries that don't store one.
        self.mod._save([
            {"id": "a", "name": "A", "text": "a", "tags": []},
            {"id": "b", "name": "B", "text": "b", "tags": [], "negative": "lowres, blurry"},
        ])
        resp = asyncio.run(self.mod.list_prompts(FakeRequest()))
        body = json.loads(resp.body)
        by_id = {p["id"]: p for p in body["prompts"]}
        self.assertEqual(by_id["a"]["negative"], "")
        self.assertEqual(by_id["b"]["negative"], "lowres, blurry")

    def test_upsert_route_accepts_negative_field(self):
        req = FakeRequest(post_data={"name": "Cat", "text": "fluffy",
                                       "negative": "no_dogs, no_cars"})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        body = json.loads(resp.body)
        self.assertEqual(resp.status, 200)
        items = self.mod._load()
        self.assertEqual(items[0]["negative"], "no_dogs, no_cars")

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

    def test_parse_tags_string_preserves_case_with_ci_dedupe(self):
        # Case-preserving: keeps the user's `Cards:beast` capitalization.
        # Case-insensitive dedupe: drops the trailing lowercase "character"
        # duplicate but keeps the first occurrence "Character".
        self.assertEqual(self.mod._parse_tags("Character, Fantasy, character"),
                         ["Character", "Fantasy"])

    def test_parse_tags_list(self):
        self.assertEqual(self.mod._parse_tags(["Sci-Fi", "Sci-Fi", "  noir "]),
                         ["Sci-Fi", "noir"])

    def test_parse_tags_preserves_prefix_capitalization(self):
        self.assertEqual(
            self.mod._parse_tags(["Cards", "Cards:beast", "cards:beast"]),
            ["Cards", "Cards:beast"],
        )

    def test_parse_tags_none_or_empty(self):
        self.assertEqual(self.mod._parse_tags(None), [])
        self.assertEqual(self.mod._parse_tags(""), [])
        self.assertEqual(self.mod._parse_tags(",,, ,"), [])

    def test_upsert_persists_tags(self):
        req = FakeRequest(post_data={"name": "Knight", "text": "armor", "tags": "Character, Fantasy"})
        body = json.loads(asyncio.run(self.mod.upsert_prompt(req)).body)
        self.assertEqual(body["tags"], ["Character", "Fantasy"])
        items = self.mod._load()
        self.assertEqual(items[0]["tags"], ["Character", "Fantasy"])

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
        self.assertEqual(entry["tags"], ["Character", "Fantasy"])

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

    def test_import_csv_accepts_optional_negative_column(self):
        text = "name,text,negative\nKnight,a knight,lowres bad_anatomy\n"
        added, _, _, errors = self.mod._import_csv(text)
        self.assertEqual(added, 1)
        self.assertEqual(errors, [])
        items = self.mod._load()
        self.assertEqual(items[0]["negative"], "lowres bad_anatomy")

    def test_import_csv_omits_negative_field_when_column_absent(self):
        # No negative column at all — the existing CSV format keeps working
        # and entries don't get a stray empty negative field.
        text = "name,text\nKnight,a knight\n"
        added, _, _, errors = self.mod._import_csv(text)
        self.assertEqual(added, 1)
        self.assertEqual(errors, [])
        self.assertNotIn("negative", self.mod._load()[0])

    def test_import_csv_blank_negative_on_update_clears_existing(self):
        # Mirrors the Save-node semantics: blank input on update explicitly
        # clears any prior value.
        self.mod._save([{"id": "k", "name": "Knight", "text": "a knight",
                          "tags": [], "negative": "lowres"}])
        text = "name,text,negative,id\nKnight,a knight,,k\n"
        _, updated, _, _ = self.mod._import_csv(text, mode="update")
        self.assertEqual(updated, 1)
        self.assertEqual(self.mod._load()[0]["negative"], "")

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
        items = []
        for p in prompts:
            entry = {"id": p["id"], "name": p.get("name", p["id"]),
                     "text": p.get("text", ""), "tags": p.get("tags", [])}
            if "negative" in p:
                entry["negative"] = p["negative"]
            items.append(entry)
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
        prompt, action, seed, count, negative = node.assemble(
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
        self.assertEqual(negative, "")  # no negatives wired

    def test_comic_frame_clamps_index_above_range(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, seed, count, _neg = node.assemble(
            frames_json='["a", "b"]', frame_index=99,
            character="x",
        )
        self.assertEqual(action, "b")  # clamped to last frame
        self.assertEqual(prompt, "x, b")

    def test_comic_frame_clamps_index_below_range(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, _seed, _count, _neg = node.assemble(
            frames_json='["a", "b"]', frame_index=0,
        )
        self.assertEqual(action, "a")

    def test_comic_frame_no_frames_emits_anchors_only(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, _seed, count, _neg = node.assemble(
            frames_json='[]', frame_index=1,
            character="hero", scene="night", background="rooftop",
        )
        self.assertEqual(prompt, "hero, night, rooftop")
        self.assertEqual(action, "")
        self.assertEqual(count, 0)

    def test_comic_frame_invalid_frames_json_falls_back(self):
        node = self.mod.PromptLibraryComicFrame()
        prompt, action, _, count, _neg = node.assemble(
            frames_json='not valid json', frame_index=1, character="x",
        )
        self.assertEqual(prompt, "x")
        self.assertEqual(action, "")
        self.assertEqual(count, 0)

    def test_comic_frame_joins_wired_negatives(self):
        node = self.mod.PromptLibraryComicFrame()
        _prompt, _action, _seed, _count, negative = node.assemble(
            frames_json='["a"]', frame_index=1, separator=", ",
            character="hero",
            character_negative="lowres",
            scene_negative="bad_anatomy",
            background_negative="watermark",
        )
        self.assertEqual(negative, "lowres, bad_anatomy, watermark")

    def test_comic_frame_skips_blank_negatives_in_join(self):
        node = self.mod.PromptLibraryComicFrame()
        _, _, _, _, negative = node.assemble(
            frames_json='["a"]', frame_index=1, separator=", ",
            character_negative="lowres",
            scene_negative="",            # blank — skipped
            background_negative="watermark",
        )
        self.assertEqual(negative, "lowres, watermark")

    def test_comic_frame_seed_offset_per_frame(self):
        node = self.mod.PromptLibraryComicFrame()
        seeds = []
        for i in range(1, 4):
            _, _, seed, _, _ = node.assemble(
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
        text, pid, neg = node.pick(tag_filter="character", seed=0)
        self.assertIn(pid, {"elf", "wizard"})
        self.assertEqual(text, pid)
        self.assertEqual(neg, "")

    def test_random_and_filter_multi_tag(self):
        self._seed_library([
            {"id": "elf", "text": "elf", "tags": ["character", "fantasy"]},
            {"id": "wizard", "text": "wizard", "tags": ["character", "fantasy"]},
            {"id": "robot", "text": "robot", "tags": ["character", "scifi"]},
        ])
        node = self.mod.PromptLibraryRandom()
        text, pid, _ = node.pick(tag_filter="character, fantasy", seed=0)
        self.assertIn(pid, {"elf", "wizard"})

    def test_random_no_match_returns_empty(self):
        self._seed_library([{"id": "x", "text": "x", "tags": ["a"]}])
        node = self.mod.PromptLibraryRandom()
        out = node.pick(tag_filter="missing", seed=0)
        self.assertEqual(out, ("", "", ""))

    def test_random_empty_filter_picks_anything(self):
        self._seed_library([
            {"id": "a", "text": "a"},
            {"id": "b", "text": "b"},
        ])
        node = self.mod.PromptLibraryRandom()
        _, pid, _ = node.pick(tag_filter="", seed=0)
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
        text, pid, _ = node.pick(tag_filter="character", seed=0, expand_wildcards=True)
        self.assertIn(text, {"red knight", "blue knight"})
        self.assertEqual(pid, "knight")

    def test_random_emits_negative_when_present(self):
        self._seed_library([
            {"id": "k", "text": "knight", "tags": ["c"], "negative": "lowres, blurry"},
        ])
        node = self.mod.PromptLibraryRandom()
        text, pid, neg = node.pick(tag_filter="c", seed=0)
        self.assertEqual((text, pid, neg), ("knight", "k", "lowres, blurry"))

    def test_random_expands_wildcards_in_negative(self):
        self._seed_library([
            {"id": "k", "text": "knight", "tags": ["c"],
             "negative": "{lowres|blurry}, watermark"},
        ])
        node = self.mod.PromptLibraryRandom()
        _, _, neg = node.pick(tag_filter="c", seed=0, expand_wildcards=True)
        self.assertIn(neg, {"lowres, watermark", "blurry, watermark"})

    # ---- PromptLibrarySave with negative -------------------------------

    def test_save_node_stores_negative_when_provided(self):
        node = self.mod.PromptLibrarySave()
        node.save(name="Knight", text="knight", negative="lowres, blurry")
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "knight")
        self.assertEqual(items[0]["negative"], "lowres, blurry")

    def test_save_node_skips_negative_when_blank(self):
        # Blank negative on a fresh entry means: don't even write the field,
        # to keep prompts.json tidy for the 95% of entries that don't use it.
        node = self.mod.PromptLibrarySave()
        node.save(name="Knight", text="knight")
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertNotIn("negative", items[0])

    def test_save_node_clears_negative_when_explicitly_blanked(self):
        # If the entry already had a negative and the user updates with blank,
        # the field should be set to "" (cleared), not silently retained.
        node = self.mod.PromptLibrarySave()
        node.save(name="Knight", text="knight", negative="lowres")
        node.save(name="Knight", text="knight", negative="", overwrite_by_name=True)
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get("negative", "MISSING"), "")

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

    def test_import_zip_rejects_zip_bomb_via_uncompressed_cap(self):
        # Zip-bomb defence: sum of `info.file_size` across the central
        # directory is checked before any member is decompressed. Build a
        # legit small zip and shrink the cap to a value below it so the
        # guard fires deterministically without a multi-GB fixture.
        items = [{"id": "abc", "name": "n", "text": "t" * 200}]
        zip_bytes = self.mod._build_export_zip(items, "test")
        original_cap = self.mod._MAX_IMPORT_ZIP_UNCOMPRESSED_BYTES
        self.mod._MAX_IMPORT_ZIP_UNCOMPRESSED_BYTES = 16  # cap below manifest size
        try:
            added, updated, skipped, errors = self.mod._import_zip(zip_bytes)
        finally:
            self.mod._MAX_IMPORT_ZIP_UNCOMPRESSED_BYTES = original_cap
        self.assertEqual((added, updated, skipped), (0, 0, 0))
        self.assertTrue(any("zip-bomb guard" in e for e in errors), errors)

    def test_import_zip_route_rejects_oversized_compressed_upload(self):
        # The route also has a compressed-bytes cap so a huge multipart
        # upload is rejected before the inner _import_zip tries to parse it.
        items = [{"id": "abc", "name": "n", "text": "t"}]
        zip_bytes = self.mod._build_export_zip(items, "test")
        original_cap = self.mod._MAX_IMPORT_ZIP_BYTES
        self.mod._MAX_IMPORT_ZIP_BYTES = 1
        try:
            req = FakeRequest(post_data={"file": FakeFileField("x.zip", zip_bytes)})
            resp = asyncio.run(self.mod.import_zip_route(req))
        finally:
            self.mod._MAX_IMPORT_ZIP_BYTES = original_cap
        self.assertEqual(resp.status, 400)
        self.assertIn(b"zip too large", resp.body)

    def test_import_csv_route_rejects_oversized_body(self):
        # CSV uploads are capped consistent with image and zip uploads;
        # without this guard a multi-GB CSV would parse into memory.
        body = "name,prompt\nfoo,hello\n"
        original_cap = self.mod._MAX_IMPORT_CSV_BYTES
        self.mod._MAX_IMPORT_CSV_BYTES = 5
        try:
            req = FakeRequest(post_data={"csv": body})
            resp = asyncio.run(self.mod.import_csv_route(req))
        finally:
            self.mod._MAX_IMPORT_CSV_BYTES = original_cap
        self.assertEqual(resp.status, 400)
        self.assertIn(b"CSV too large", resp.body)

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

    # ---- loras roundtrip -----------------------------------------------

    def test_parse_loras_accepts_json_string(self):
        raw = json.dumps([
            {"name": "Anima/Anima Turbo LoRA.safetensors", "strength_model": 0.85,
             "strength_clip": 0.85, "triggers": "anima style", "enabled": True},
            {"name": "broken (no name)".replace("broken (no name)", ""),  # → empty → dropped
             "strength_model": 1.0},
        ])
        out = self.mod._parse_loras(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Anima/Anima Turbo LoRA.safetensors")
        self.assertAlmostEqual(out[0]["strength_model"], 0.85)
        self.assertEqual(out[0]["triggers"], "anima style")

    def test_parse_loras_caps_at_ten(self):
        loras = [{"name": f"l{i}.safetensors"} for i in range(15)]
        out = self.mod._parse_loras(loras)
        self.assertEqual(len(out), 10)

    def test_parse_loras_clamps_strength(self):
        out = self.mod._parse_loras([{"name": "x.safetensors", "strength_model": 99}])
        self.assertEqual(out[0]["strength_model"], 2.0)
        self.assertEqual(out[0]["strength_clip"], 2.0)

    def test_parse_loras_invalid_input_returns_empty(self):
        self.assertEqual(self.mod._parse_loras(None), [])
        self.assertEqual(self.mod._parse_loras(""), [])
        self.assertEqual(self.mod._parse_loras("not json"), [])
        self.assertEqual(self.mod._parse_loras({"not": "a list"}), [])

    def test_upsert_persists_loras(self):
        loras = json.dumps([
            {"name": "Anima/Anima.safetensors", "strength_model": 0.7,
             "strength_clip": 0.7, "triggers": "anima"},
            {"name": "char/Hatsune.safetensors", "strength_model": 1.0,
             "strength_clip": 1.0, "triggers": ""},
        ])
        req = FakeRequest(post_data={"name": "Style A", "text": "fluffy", "loras": loras})
        resp = asyncio.run(self.mod.upsert_prompt(req))
        body = json.loads(resp.body)
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(body["loras"]), 2)
        self.assertEqual(body["loras"][0]["name"], "Anima/Anima.safetensors")
        items = self.mod._load()
        self.assertEqual(len(items[0]["loras"]), 2)

    def test_upsert_omitting_loras_preserves_existing(self):
        loras = json.dumps([{"name": "x.safetensors", "strength_model": 1.0}])
        req1 = FakeRequest(post_data={"name": "S", "text": "t", "loras": loras})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        # Edit name + text but DON'T send loras — old payload should be preserved.
        req2 = FakeRequest(post_data={"id": body1["id"], "name": "S", "text": "v2"})
        body2 = json.loads(asyncio.run(self.mod.upsert_prompt(req2)).body)
        self.assertEqual(len(body2["loras"]), 1)
        self.assertEqual(body2["loras"][0]["name"], "x.safetensors")

    def test_upsert_empty_loras_clears(self):
        loras = json.dumps([{"name": "x.safetensors"}])
        req1 = FakeRequest(post_data={"name": "S", "text": "t", "loras": loras})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        # Now send loras="[]" — backend should clear the list.
        req2 = FakeRequest(post_data={"id": body1["id"], "name": "S", "text": "t",
                                       "loras": "[]"})
        body2 = json.loads(asyncio.run(self.mod.upsert_prompt(req2)).body)
        self.assertEqual(body2["loras"], [])

    def test_history_captures_lora_change(self):
        l1 = json.dumps([{"name": "a.safetensors", "strength_model": 1.0}])
        req1 = FakeRequest(post_data={"name": "S", "text": "same", "loras": l1})
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(req1)).body)
        l2 = json.dumps([{"name": "b.safetensors", "strength_model": 1.0}])
        req2 = FakeRequest(post_data={"id": body1["id"], "name": "S", "text": "same",
                                       "loras": l2})
        asyncio.run(self.mod.upsert_prompt(req2))
        items = self.mod._load()
        # Only the loras changed — name + text stayed same. History should still
        # have a snapshot recording the prior loras.
        self.assertEqual(len(items[0].get("history") or []), 1)
        self.assertEqual(items[0]["history"][0]["loras"][0]["name"], "a.safetensors")

    def test_loras_route_returns_list(self):
        # folder_paths may be None in the test env (no ComfyUI on path); the
        # route should degrade gracefully rather than 500.
        resp = asyncio.run(self.mod.list_loras(FakeRequest()))
        body = json.loads(resp.body)
        self.assertIn("loras", body)
        self.assertIsInstance(body["loras"], list)

    def test_list_route_includes_loras_field(self):
        loras = json.dumps([{"name": "a.safetensors", "strength_model": 1.0}])
        asyncio.run(self.mod.upsert_prompt(
            FakeRequest(post_data={"name": "Z", "text": "t", "loras": loras})))
        resp = asyncio.run(self.mod.list_prompts(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(len(body["prompts"]), 1)
        self.assertEqual(len(body["prompts"][0]["loras"]), 1)

    def test_duplicate_copies_loras(self):
        loras = json.dumps([{"name": "a.safetensors", "strength_model": 0.5}])
        body1 = json.loads(asyncio.run(self.mod.upsert_prompt(
            FakeRequest(post_data={"name": "Orig", "text": "t", "loras": loras}))).body)
        resp = asyncio.run(self.mod.duplicate_prompt(
            FakeRequest(json_data={"id": body1["id"]})))
        new_id = json.loads(resp.body)["id"]
        items = {i["id"]: i for i in self.mod._load()}
        self.assertEqual(len(items[new_id]["loras"]), 1)
        self.assertEqual(items[new_id]["loras"][0]["name"], "a.safetensors")

    def test_style_format_prompt_combines_text_and_triggers(self):
        # style_node.py imports torch + comfy.sd which aren't present in
        # .testenv. Stub them, register the test-loaded package so its
        # `from . import _load, _lock` resolves, then import style_node.
        import sys, types, importlib.util
        stubs = ["torch", "comfy", "comfy.sd", "comfy.utils", "folder_paths"]
        installed = []
        pkg_name = self.mod.__name__   # "plib_<tmp>"
        try:
            for name in stubs:
                if name not in sys.modules:
                    sys.modules[name] = types.ModuleType(name)
                    installed.append(name)
            # Register the test module as a package and register it under the
            # name `from . import` will look up.
            sys.modules[pkg_name] = self.mod
            self.mod.__path__ = [str(NODE_DIR)]
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.style_node", NODE_DIR / "style_node.py")
            style_node = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg_name}.style_node"] = style_node
            spec.loader.exec_module(style_node)
            loras = [
                {"name": "a", "triggers": "anime style", "enabled": True},
                {"name": "b", "triggers": "blue eyes", "enabled": False},
                {"name": "c", "triggers": "  ", "enabled": True},
                {"name": "d", "triggers": "neon, vapor", "enabled": True},
            ]
            self.assertEqual(style_node._format_prompt("a girl", loras),
                              "a girl, anime style, neon, vapor")
            self.assertEqual(style_node._format_prompt("", loras),
                              "anime style, neon, vapor")
            self.assertEqual(style_node._format_prompt("only text", []), "only text")
            # Trailing commas + whitespace get cleaned so the join doesn't
            # produce a double-comma artefact.
            self.assertEqual(
                style_node._format_prompt("a girl, ",
                    [{"name": "a", "triggers": "blue eyes,", "enabled": True}]),
                "a girl, blue eyes",
            )
            # extra_text appends after triggers, no leading comma when text + triggers empty.
            self.assertEqual(
                style_node._format_prompt("base", loras, "moody lighting"),
                "base, anime style, neon, vapor, moody lighting",
            )
            self.assertEqual(
                style_node._format_prompt("", [], "just extras"), "just extras")
        finally:
            for name in installed:
                sys.modules.pop(name, None)
            sys.modules.pop(f"{pkg_name}.style_node", None)
            sys.modules.pop(pkg_name, None)

    # ---- library validator route + fix_orphans -------------------------

    def test_validate_route_reports_clean_library(self):
        self.mod._save([{"id": "k1", "name": "K1", "text": "v1"},
                          {"id": "k2", "name": "K2", "text": "v2"}])
        resp = asyncio.run(self.mod.validate_library(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(body["entries_count"], 2)
        self.assertEqual(body["thumbnails_count"], 0)
        self.assertEqual(body["broken_loras"], [])
        self.assertEqual(body["orphan_images"], [])
        self.assertEqual(body["empty_texts"], [])
        self.assertEqual(body["invalid_ids"], [])

    def test_validate_route_flags_empty_texts(self):
        self.mod._save([{"id": "ok", "name": "OK", "text": "value"},
                          {"id": "empty", "name": "Bad", "text": ""}])
        body = json.loads(asyncio.run(self.mod.validate_library(FakeRequest())).body)
        self.assertEqual(len(body["empty_texts"]), 1)
        self.assertEqual(body["empty_texts"][0]["id"], "empty")

    def test_validate_route_finds_orphan_thumbnails(self):
        self.mod._save([{"id": "k1", "name": "K1", "text": "v"}])
        # Drop a stray thumbnail in IMAGES_DIR for an id that no entry has.
        self.mod.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        (self.mod.IMAGES_DIR / "ghost.png").write_bytes(_real_png())
        body = json.loads(asyncio.run(self.mod.validate_library(FakeRequest())).body)
        self.assertIn("ghost", body["orphan_images"])

    def test_fix_orphans_removes_stray_thumbnails(self):
        self.mod._save([{"id": "k1", "name": "K1", "text": "v"}])
        self.mod.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        (self.mod.IMAGES_DIR / "ghost.png").write_bytes(_real_png())
        (self.mod.IMAGES_DIR / "phantom.jpg").write_bytes(_real_png())
        resp = asyncio.run(self.mod.fix_orphans(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(body["removed"], 2)
        # Re-validate: orphans gone.
        after = json.loads(asyncio.run(self.mod.validate_library(FakeRequest())).body)
        self.assertEqual(after["orphan_images"], [])

    # ---- Save node loras_json ------------------------------------------

    def test_save_overwrite_by_name_picks_most_recently_updated(self):
        """When multiple entries share a name, overwrite_by_name should
        target the most-recently-updated one — not the first by storage
        order. Picking-the-oldest was confusing when users expected
        'the one I last edited' semantics."""
        # Seed two entries with the same name, different updated_at.
        self.mod._save([
            {"id": "old_dup", "name": "Foo", "text": "old version",
             "tags": [], "updated_at": 100.0},
            {"id": "new_dup", "name": "Foo", "text": "newer version",
             "tags": [], "updated_at": 999.0},
        ])
        node = self.mod.PromptLibrarySave()
        text_out, eid = node.save(name="Foo", text="updated body",
                                    overwrite_by_name=True)
        # Should have updated the newer one, not the older.
        self.assertEqual(eid, "new_dup")
        items = {i["id"]: i for i in self.mod._load()}
        self.assertEqual(items["new_dup"]["text"], "updated body")
        self.assertEqual(items["old_dup"]["text"], "old version")  # untouched

    def test_save_overwrite_by_name_creates_when_no_match(self):
        """overwrite_by_name with no existing entry of that name still
        creates a new entry — no behaviour change from before."""
        node = self.mod.PromptLibrarySave()
        text_out, eid = node.save(name="Brand new", text="hi",
                                    overwrite_by_name=True)
        self.assertTrue(eid)
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "Brand new")

    def test_save_node_persists_loras_json(self):
        """PromptLibrarySave grew a loras_json input — when set, the saved
        entry carries the LoRA stack just as if it had been edited via
        the modal. Empty string leaves the entry's loras untouched."""
        node = self.mod.PromptLibrarySave()
        loras_json = json.dumps([
            {"name": "Anima/Anima.safetensors", "strength_model": 0.85,
             "strength_clip": 0.85, "triggers": "anime", "enabled": True},
        ])
        text_out, eid = node.save(name="Saved with loras",
                                    text="masterpiece",
                                    loras_json=loras_json)
        items = self.mod._load()
        e = next(i for i in items if i["id"] == eid)
        self.assertEqual(len(e["loras"]), 1)
        self.assertEqual(e["loras"][0]["name"], "Anima/Anima.safetensors")

    def test_save_node_empty_loras_json_leaves_existing_alone(self):
        # Pre-seed an entry with loras.
        self.mod._save([{"id": "k1", "name": "K", "text": "v",
                          "loras": [{"name": "x.safetensors",
                                       "strength_model": 1.0,
                                       "strength_clip": 1.0,
                                       "triggers": "", "enabled": True}]}])
        node = self.mod.PromptLibrarySave()
        # Empty loras_json: must NOT clear the existing list.
        node.save(name="K", text="v2", prompt_id="k1", loras_json="")
        items = self.mod._load()
        self.assertEqual(len(items[0]["loras"]), 1)
        self.assertEqual(items[0]["loras"][0]["name"], "x.safetensors")

    def test_save_node_explicit_empty_list_clears_loras(self):
        self.mod._save([{"id": "k1", "name": "K", "text": "v",
                          "loras": [{"name": "x.safetensors",
                                       "strength_model": 1.0,
                                       "strength_clip": 1.0,
                                       "triggers": "", "enabled": True}]}])
        node = self.mod.PromptLibrarySave()
        node.save(name="K", text="v2", prompt_id="k1", loras_json="[]")
        items = self.mod._load()
        self.assertEqual(items[0]["loras"], [])

    # ---- end-to-end round-trip: save → export zip → import zip --------

    def test_export_import_roundtrip_preserves_full_entry(self):
        """Save an entry with every supported field, export, wipe, reimport,
        verify nothing was lost. Catches silent data loss whenever the
        export manifest or import path forgets a field."""
        png_bytes = _real_png()
        loras = json.dumps([
            {"name": "Anima/Anima.safetensors", "strength_model": 0.85,
             "strength_clip": 0.85, "triggers": "anime style", "enabled": True},
            {"name": "char/x.safetensors", "strength_model": 1.0,
             "strength_clip": 1.0, "triggers": "", "enabled": False},
        ])
        # 1. Save a complex entry.
        save_req = FakeRequest(post_data={
            "name": "Round-trip target",
            "text": "fluffy cat, masterpiece",
            "negative": "lowres, bad_anatomy",
            "tags": "character, model:anima",
            "rating": "4",
            "notes": "Roundtrip test entry",
            "loras": loras,
            "image": FakeFileField("rt.png", png_bytes),
        })
        body = json.loads(asyncio.run(self.mod.upsert_prompt(save_req)).body)
        pid = body["id"]

        # 2. Export — empty ids list means everything.
        export_req = FakeRequest(json_data={"ids": [pid]})
        export_resp = asyncio.run(self.mod.export_zip(export_req))
        self.assertEqual(export_resp.status, 200)
        zip_bytes = export_resp.body

        # 3. Wipe the library + thumbnail.
        self.mod._save([])
        self.mod._delete_image_files(pid)
        self.assertIsNone(self.mod._image_path_for(pid))

        # 4. Reimport.
        added, updated, skipped, errors = self.mod._import_zip(zip_bytes, mode="add_only")
        self.assertEqual(added, 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(errors, [])

        # 5. Verify every field round-tripped.
        items = self.mod._load()
        self.assertEqual(len(items), 1)
        e = items[0]
        self.assertEqual(e["id"], pid)
        self.assertEqual(e["name"], "Round-trip target")
        self.assertEqual(e["text"], "fluffy cat, masterpiece")
        self.assertEqual(e["negative"], "lowres, bad_anatomy")
        self.assertIn("character", e["tags"])
        self.assertIn("model:anima", e["tags"])
        self.assertEqual(e["rating"], 4)
        self.assertEqual(e["notes"], "Roundtrip test entry")
        self.assertEqual(len(e["loras"]), 2)
        self.assertEqual(e["loras"][0]["name"], "Anima/Anima.safetensors")
        self.assertAlmostEqual(e["loras"][0]["strength_model"], 0.85)
        self.assertEqual(e["loras"][0]["triggers"], "anime style")
        self.assertEqual(e["loras"][1]["enabled"], False)
        # Thumbnail file should also have been restored.
        self.assertIsNotNone(self.mod._image_path_for(pid))

    def test_export_import_roundtrip_add_only_skips_existing(self):
        """add_only mode must NOT overwrite an existing entry — protects
        local edits when reimporting a shared zip. The skipped entry's
        in-library text/notes/etc. stay intact."""
        loras_json = json.dumps([{"name": "a.safetensors", "strength_model": 1.0}])
        body = json.loads(asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={
            "name": "shared", "text": "v1", "loras": loras_json,
        }))).body)
        pid = body["id"]

        export_resp = asyncio.run(self.mod.export_zip(
            FakeRequest(json_data={"ids": [pid]})))
        zip_bytes = export_resp.body

        # Mutate locally — different text, no loras.
        asyncio.run(self.mod.upsert_prompt(FakeRequest(post_data={
            "id": pid, "name": "shared", "text": "v2 (local edit)",
            "loras": "[]",
        })))

        added, updated, skipped, errors = self.mod._import_zip(zip_bytes, mode="add_only")
        self.assertEqual(skipped, 1)
        self.assertEqual(added, 0)
        items = self.mod._load()
        self.assertEqual(items[0]["text"], "v2 (local edit)")
        self.assertEqual(items[0].get("loras", []), [])

    # ---- import_backgrounds + import_tag_packs routes ------------------

    def test_import_backgrounds_route_creates_entries(self):
        """import_backgrounds walks the BG_PRESETS list (excluding the
        '(none)' sentinel + the divider rows) and seeds the library."""
        before = len(self.mod._load())
        resp = asyncio.run(self.mod.import_backgrounds_route(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(resp.status, 200)
        self.assertGreater(body["added"], 0)
        items = self.mod._load()
        self.assertEqual(len(items) - before, body["added"])
        # Every imported entry has the 'location' tag.
        for entry in items[before:]:
            self.assertIn("location", entry["tags"])

    def test_import_backgrounds_route_idempotent_without_refresh(self):
        """Re-running with refresh_existing=False should report all skipped
        instead of duplicating entries (slug collision protection)."""
        asyncio.run(self.mod.import_backgrounds_route(FakeRequest()))
        first_count = len(self.mod._load())
        resp2 = asyncio.run(self.mod.import_backgrounds_route(
            FakeRequest(json_data={"refresh_existing": False})))
        body = json.loads(resp2.body)
        self.assertEqual(body["added"], 0)
        self.assertGreater(body["skipped"], 0)
        self.assertEqual(len(self.mod._load()), first_count)

    def test_import_tag_packs_rejects_non_zip(self):
        """Sanity-check the basic error path for the route — passing a
        non-zip body returns the underlying _import_tag_pack_zip error
        without crashing."""
        # No file field at all.
        resp = asyncio.run(self.mod.import_tag_packs_route(FakeRequest()))
        body = json.loads(resp.body)
        self.assertEqual(resp.status, 400)
        self.assertIn("error", body)

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
