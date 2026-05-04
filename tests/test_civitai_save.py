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


class WorkflowExtractionTests(unittest.TestCase):
    def test_extracts_checkpoint_and_lora_chain(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "anima_v4.safetensors"}},
            "2": {"class_type": "LoraLoader",
                   "inputs": {"lora_name": "detail.safetensors",
                              "strength_model": 0.8, "strength_clip": 0.8,
                              "model": ["1", 0], "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "a cat", "clip": ["2", 1]}},
            "4": {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "bad anatomy", "clip": ["2", 1]}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": 42, "steps": 24, "cfg": 7.0,
                              "sampler_name": "dpmpp_2m", "scheduler": "karras",
                              "model": ["2", 0],
                              "positive": ["3", 0], "negative": ["4", 0],
                              "latent_image": ["x", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["model_label"], "checkpoints::anima_v4.safetensors")
        self.assertEqual(meta["loras"], [("detail.safetensors", 0.8)])
        self.assertEqual(meta["positive"], "a cat")
        self.assertEqual(meta["negative"], "bad anatomy")
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["steps"], 24)
        self.assertEqual(meta["cfg"], 7.0)
        self.assertEqual(meta["sampler_name"], "dpmpp_2m")
        self.assertEqual(meta["scheduler"], "karras")

    def test_extracts_unet_loader_for_diffusion_models(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "UNETLoader",
                   "inputs": {"unet_name": "anima_pony.safetensors"}},
            "9": {"class_type": "KSampler",
                   "inputs": {"model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["model_label"], "diffusion_models::anima_pony.safetensors")

    def test_extracts_rgthree_power_lora_loader(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
                "model": ["1", 0], "clip": ["1", 1],
                "lora_1": {"on": True, "lora": "first.safetensors", "strength": 0.6},
                "lora_2": {"on": False, "lora": "skipped.safetensors", "strength": 1.0},
                "lora_3": {"on": True, "lora": "second.safetensors", "strength": 1.2,
                           "strengthTwo": 0.5},
            }},
            "9": {"class_type": "KSampler",
                   "inputs": {"model": ["2", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["model_label"], "checkpoints::model.safetensors")
        self.assertEqual(meta["loras"],
                          [("first.safetensors", 0.6), ("second.safetensors", 1.2)])

    def test_picks_last_ksampler(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "5": {"class_type": "KSampler",
                   "inputs": {"seed": 11, "steps": 10, "cfg": 5.0,
                              "sampler_name": "euler", "scheduler": "normal",
                              "model": ["1", 0]}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": 99, "steps": 30, "cfg": 8.0,
                              "sampler_name": "dpmpp_3m_sde", "scheduler": "karras",
                              "model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["seed"], 99)
        self.assertEqual(meta["steps"], 30)
        self.assertEqual(meta["sampler_name"], "dpmpp_3m_sde")

    def test_empty_prompt_returns_empty_dict(self):
        from civitai_save import extract_workflow_metadata
        self.assertEqual(extract_workflow_metadata(None), {})
        self.assertEqual(extract_workflow_metadata({}), {})

    def test_efficiency_sdxl_sampler_via_pack_tuple(self):
        # The user's high-rez workflow shape: Eff. SDXL sampler reads model /
        # positive / negative through `sdxl_tuple` -> Pack SDXL Tuple ->
        # ImpactWildcardEncode -> PowerLoraLoader -> CheckpointLoaderSimple.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "2": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "krakenNOIR_v3.safetensors"}},
            "113": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
                "model": ["2", 0], "clip": ["2", 1],
                "lora_1": {"on": True, "lora": "EldritchComicsXL1.2.safetensors",
                           "strength": 0.5},
                "lora_2": {"on": True, "lora": "InkArtXL_1.2.safetensors",
                           "strength": 0.5},
            }},
            "56": {"class_type": "ImpactWildcardEncode", "inputs": {
                "wildcard_text": "(Solo:1.2), __character__, ponytail",
                "populated_text": "(Solo:1.2), brunette female, ponytail",
                "model": ["113", 0], "clip": ["113", 1],
            }},
            "101": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": "bad anatomy", "clip": ["56", 1]}},
            "189": {"class_type": "Pack SDXL Tuple", "inputs": {
                "base_model": ["56", 0], "base_clip": ["56", 1],
                "base_positive": ["56", 2], "base_negative": ["101", 0],
            }},
            "4": {"class_type": "KSampler SDXL (Eff.)", "inputs": {
                "noise_seed": 311603508439804, "steps": 30, "cfg": 5.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                "sdxl_tuple": ["189", 0],
            }},
        }
        m = extract_workflow_metadata(prompt)
        self.assertEqual(m["model_label"], "checkpoints::krakenNOIR_v3.safetensors")
        self.assertEqual(m["loras"], [
            ("EldritchComicsXL1.2.safetensors", 0.5),
            ("InkArtXL_1.2.safetensors", 0.5),
        ])
        # populated_text wins over wildcard_text (resolved value).
        self.assertEqual(m["positive"], "(Solo:1.2), brunette female, ponytail")
        self.assertEqual(m["negative"], "bad anatomy")
        self.assertEqual(m["seed"], 311603508439804)
        self.assertEqual(m["steps"], 30)
        self.assertEqual(m["sampler_name"], "dpmpp_2m")
        self.assertEqual(m["scheduler"], "karras")

    def test_rgthree_lora_loader_stack(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "Lora Loader Stack (rgthree)", "inputs": {
                "model": ["1", 0], "clip": ["1", 1],
                "lora_01": "first.safetensors", "strength_01": 0.7,
                "lora_02": "None", "strength_02": 1.0,
                "lora_03": "third.safetensors", "strength_03": 0.5,
                "lora_04": "fourth.safetensors", "strength_04": 0,
            }},
            "9": {"class_type": "KSampler",
                   "inputs": {"model": ["2", 0]}},
        }
        m = extract_workflow_metadata(prompt)
        # Slot 02 ("None") and slot 04 (zero strength) are skipped.
        self.assertEqual(m["loras"],
                          [("first.safetensors", 0.7), ("third.safetensors", 0.5)])

    def test_easy_use_pipe_sampler_with_full_loader(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "easy fullLoader", "inputs": {
                "ckpt_name": "model.safetensors",
                "lora_name": "detail.safetensors",
                "lora_model_strength": 0.8, "lora_clip_strength": 0.8,
                "positive": "masterpiece anime girl",
                "negative": "low quality",
            }},
            "9": {"class_type": "easy fullkSampler", "inputs": {
                "pipe": ["1", 0],
                "seed": 12345, "steps": 25, "cfg": 6.5,
                "sampler_name": "dpmpp_2m_sde", "scheduler": "karras",
            }},
        }
        m = extract_workflow_metadata(prompt)
        self.assertEqual(m["model_label"], "checkpoints::model.safetensors")
        self.assertEqual(m["loras"], [("detail.safetensors", 0.8)])
        self.assertEqual(m["positive"], "masterpiece anime girl")
        self.assertEqual(m["negative"], "low quality")
        self.assertEqual(m["seed"], 12345)
        self.assertEqual(m["steps"], 25)
        self.assertEqual(m["sampler_name"], "dpmpp_2m_sde")

    def test_pysssss_string_function_concatenates(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "10": {"class_type": "StringFunction|pysssss", "inputs": {
                "action": "append", "tidy_tags": "yes",
                "text_a": "anime girl", "text_b": "blue hair", "text_c": "smile",
            }},
            "11": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": ["10", 0], "clip": ["1", 1]}},
            "12": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": "bad", "clip": ["1", 1]}},
            "9": {"class_type": "KSampler",
                   "inputs": {"model": ["1", 0],
                              "positive": ["11", 0], "negative": ["12", 0]}},
        }
        m = extract_workflow_metadata(prompt)
        self.assertEqual(m["positive"], "anime girl, blue hair, smile")
        self.assertEqual(m["negative"], "bad")

    def test_passthrough_through_modelsamplingdiscrete(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "ModelSamplingDiscrete",
                   "inputs": {"sampling": "v_prediction", "model": ["1", 0]}},
            "9": {"class_type": "KSampler",
                   "inputs": {"model": ["2", 0]}},
        }
        m = extract_workflow_metadata(prompt)
        self.assertEqual(m["model_label"], "checkpoints::model.safetensors")


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
