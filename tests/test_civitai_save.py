import json
import unittest
from pathlib import Path

from civitai_save import build_a1111_parameters


class A1111ParametersTests(unittest.TestCase):
    def test_minimum_fields(self):
        out = build_a1111_parameters(
            positive="a cat",
            negative="",
            width=1024, height=768,
            steps=None, sampler_name=None, scheduler=None, cfg=None, seed=None,
            model_name=None, model_sha256=None, loras=[],
        )
        self.assertIn("a cat", out)
        self.assertIn("Size: 1024x768", out)
        self.assertNotIn("Negative prompt", out)
        self.assertNotIn("Hashes:", out)

    def test_full_payload_with_civitai_hash_block(self):
        out = build_a1111_parameters(
            positive="masterpiece, anime girl",
            negative="bad anatomy",
            width=1024, height=1024,
            steps=24, sampler_name="dpmpp_2m", scheduler="karras",
            cfg=7.0, seed=12345,
            model_name="anima_pony_4.0.safetensors",
            model_sha256="0123456789abcdef" * 4,
            loras=[("add_detail_xl.safetensors", "fedcba9876543210" * 4, 0.6)],
        )
        # First block: positive prompt (with the lora token appended)
        self.assertIn("masterpiece, anime girl", out)
        self.assertIn("<lora:add_detail_xl:0.6>", out)
        # Second block: negative
        self.assertIn("\nNegative prompt: bad anatomy\n", out)
        # Third block: settings line
        self.assertIn("Steps: 24", out)
        self.assertIn("Sampler: dpmpp_2m karras", out)
        self.assertIn("CFG scale: 7", out)
        self.assertIn("Seed: 12345", out)
        self.assertIn("Size: 1024x1024", out)
        self.assertIn("Model hash: 0123456789", out)
        self.assertIn("Model: anima_pony_4.0", out)  # extension stripped
        # Hashes JSON block — Civitai uses this for resource matching.
        hashes_idx = out.index("Hashes: ")
        hashes_json = out[hashes_idx + len("Hashes: "):].split("\n", 1)[0]
        # Trim trailing comma-segment if fields continue (they don't, here).
        decoded = json.loads(hashes_json)
        self.assertEqual(decoded["model"], "0123456789")
        self.assertEqual(decoded["lora:add_detail_xl"], "fedcba9876")

    def test_missing_hash_skips_hash_block(self):
        out = build_a1111_parameters(
            positive="cat", negative="",
            width=512, height=512,
            steps=20, sampler_name="euler", scheduler=None,
            cfg=7.0, seed=1,
            model_name="some_model.safetensors", model_sha256=None,
            loras=[("lora.safetensors", None, 1.0)],
        )
        self.assertNotIn("Hashes:", out)
        self.assertNotIn("Model hash:", out)
        self.assertIn("Model: some_model", out)


class HashCacheTests(unittest.TestCase):
    def test_signature_changes_with_mtime(self):
        from civitai_save import _file_signature
        import tempfile, os, time
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            sig1 = _file_signature(path)
            time.sleep(1.1)
            os.utime(path, None)
            sig2 = _file_signature(path)
            self.assertIsNotNone(sig1)
            self.assertIsNotNone(sig2)
            self.assertNotEqual(sig1, sig2)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
