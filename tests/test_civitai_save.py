import json
import unittest

import time
from civitai_save import (
    build_a1111_parameters,
    _build_civitai_filename,
    _expand_filename_tokens,
    _coerce_append_counter,
    _expand_prompt_tokens,
    _extract_prompt_library_ids,
    _extract_random_pick_ids,
    _sanitize_for_filename,
    _replay_random_pick_from_items,
    _build_library_entry_snapshot,
    _normalize_lora_for_snapshot,
    _resolve_individual_pids,
)


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

    def test_grimmribbity_sdxl_sampler_via_grimmribbity_pack(self):
        # This repo's own SDXL sampler + pack node — same shape as Eff.'s but
        # both class names differ. The trace must accept both.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "krakenNOIR_v3.safetensors"}},
            "10": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
                "model": ["1", 0], "clip": ["1", 1],
                "lora_1": {"on": True, "lora": "EldritchComicsXL1.2.safetensors",
                           "strength": 0.3},
            }},
            "20": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": "noir cityscape", "clip": ["10", 1]}},
            "21": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": "watermark", "clip": ["10", 1]}},
            "30": {"class_type": "GrimmRibbityPackSDXLTuple", "inputs": {
                "base_model": ["10", 0], "base_clip": ["10", 1],
                "base_positive": ["20", 0], "base_negative": ["21", 0],
            }},
            "40": {"class_type": "GrimmRibbitySamplerSDXL", "inputs": {
                "noise_seed": 458592068423968, "steps": 30, "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde_gpu", "scheduler": "karras",
                "sdxl_tuple": ["30", 0],
            }},
        }
        m = extract_workflow_metadata(prompt)
        self.assertEqual(m["model_label"], "checkpoints::krakenNOIR_v3.safetensors")
        self.assertEqual(m["loras"], [("EldritchComicsXL1.2.safetensors", 0.3)])
        self.assertEqual(m["positive"], "noir cityscape")
        self.assertEqual(m["negative"], "watermark")
        self.assertEqual(m["seed"], 458592068423968)
        self.assertEqual(m["sampler_name"], "dpmpp_2m_sde_gpu")
        self.assertEqual(m["scheduler"], "karras")

    def test_grimmribbity_anima_sampler_direct_inputs(self):
        # Anima sampler — no tuple, just direct model/positive/negative wires
        # like a vanilla KSampler. Uses noise_seed instead of seed.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "UNETLoader",
                   "inputs": {"unet_name": "anima_v4.safetensors"}},
            "2": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": "1girl, sunset", "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode",
                    "inputs": {"text": "low quality", "clip": ["1", 1]}},
            "4": {"class_type": "GrimmRibbityAnimaSampler", "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["99", 0],
                "noise_seed": 12345, "steps": 25, "cfg": 4.0,
                "sampler_name": "euler", "scheduler": "normal",
            }},
        }
        m = extract_workflow_metadata(prompt)
        self.assertEqual(m["model_label"], "diffusion_models::anima_v4.safetensors")
        self.assertEqual(m["positive"], "1girl, sunset")
        self.assertEqual(m["negative"], "low quality")
        self.assertEqual(m["seed"], 12345)
        self.assertEqual(m["steps"], 25)
        self.assertEqual(m["cfg"], 4.0)
        self.assertEqual(m["sampler_name"], "euler")

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

    def test_seed_wired_from_rgthree_seed_node(self):
        # Common case: KSampler.seed is wired from a `Seed (rgthree)` node
        # rather than typed in directly. Previous behaviour was to silently
        # drop the seed; we now follow the connection back to the literal.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "5": {"class_type": "Seed (rgthree)",
                   "inputs": {"seed": 7777}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": ["5", 0], "steps": 25, "cfg": 7.0,
                              "sampler_name": "euler", "scheduler": "normal",
                              "model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["seed"], 7777)

    def test_seed_wired_from_easy_seed_node(self):
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "5": {"class_type": "easy seed", "inputs": {"seed": 9999}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": ["5", 0], "steps": 20, "cfg": 6.0,
                              "sampler_name": "dpm++_2m", "scheduler": "karras",
                              "model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["seed"], 9999)

    def test_advanced_sampler_noise_seed_wired(self):
        # KSamplerAdvanced uses noise_seed; if wired, follow the link.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "5": {"class_type": "Seed (rgthree)", "inputs": {"seed": 4242}},
            "9": {"class_type": "KSamplerAdvanced",
                   "inputs": {"noise_seed": ["5", 0], "steps": 30, "cfg": 5.0,
                              "sampler_name": "euler", "scheduler": "normal",
                              "model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["seed"], 4242)

    def test_seed_via_two_hops(self):
        # Seed node -> primitive passthrough -> KSampler. Two hops.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "3": {"class_type": "Seed (rgthree)", "inputs": {"seed": 1234567}},
            "4": {"class_type": "Reroute", "inputs": {"value": ["3", 0]}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": ["4", 0], "steps": 20, "cfg": 6.0,
                              "sampler_name": "euler", "scheduler": "normal",
                              "model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertEqual(meta["seed"], 1234567)

    def test_seed_unresolvable_chain_returns_no_seed(self):
        # If the chain doesn't terminate in a literal, we should drop the
        # seed gracefully — not crash, not invent a value.
        from civitai_save import extract_workflow_metadata
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "m.safetensors"}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": ["missing", 0], "steps": 20, "cfg": 6.0,
                              "sampler_name": "euler", "scheduler": "normal",
                              "model": ["1", 0]}},
        }
        meta = extract_workflow_metadata(prompt)
        self.assertNotIn("seed", meta)
        self.assertEqual(meta.get("steps"), 20)  # unaffected siblings still work

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


class WalkModelChainCycleTests(unittest.TestCase):
    """A malformed workflow forming a model-chain cycle (node A → node B
    → node A) used to silently exit the walker via the `seen` set, which
    is correct behaviour but not exercised by any test. This locks in
    that the walker terminates instead of recursing forever."""

    def test_lora_chain_with_self_referential_cycle(self):
        from civitai_save import extract_workflow_metadata
        # KSampler → LoraLoader_A.model → LoraLoader_B.model → LoraLoader_A
        # (impossible in real Comfy, but a cycle in the graph data).
        prompt = {
            "1": {"class_type": "KSampler",
                   "inputs": {"model": ["10", 0], "positive": ["20", 0],
                              "negative": ["20", 0], "seed": 1, "steps": 10,
                              "cfg": 7, "sampler_name": "euler",
                              "scheduler": "simple"}},
            "10": {"class_type": "LoraLoader",
                    "inputs": {"model": ["11", 0], "lora_name": "a.safetensors",
                                "strength_model": 1.0}},
            "11": {"class_type": "LoraLoader",
                    "inputs": {"model": ["10", 0], "lora_name": "b.safetensors",
                                "strength_model": 1.0}},
            "20": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
        }
        meta = extract_workflow_metadata(prompt)
        # Should terminate. The exact loras / model_label depend on which
        # node the walker visits first — what matters is that the call
        # returns (a finite dict) rather than recursing forever.
        self.assertIsInstance(meta, dict)


class BuildCivitaiFilenameTests(unittest.TestCase):
    """The counter toggle controls whether file names include Comfy's
    zero-padded suffix. The helper is the single source of truth — tests
    pin the four shapes the save loop can produce."""

    def test_counter_on_matches_comfy_saveimage_format(self):
        out = _build_civitai_filename("hero", 7, 0, 1, append_counter=True)
        self.assertEqual(out, "hero_00007_.png")

    def test_counter_on_pads_to_five_digits(self):
        self.assertEqual(
            _build_civitai_filename("x", 0, 0, 1, append_counter=True),
            "x_00000_.png",
        )
        self.assertEqual(
            _build_civitai_filename("x", 12345, 0, 1, append_counter=True),
            "x_12345_.png",
        )

    def test_counter_on_ignores_frame_idx_in_batch(self):
        # When the counter is on, every frame in a batch advances the counter
        # in the caller and that counter is what lands in the name. The helper
        # itself uses `counter`, not `frame_idx`, when append_counter=True.
        out = _build_civitai_filename("hero", 42, 3, 5, append_counter=True)
        self.assertEqual(out, "hero_00042_.png")

    def test_counter_off_single_frame_is_clean_name(self):
        # The exact filename the user typed (plus .png). Existing file at
        # that path WILL be overwritten by the caller's pil.save — that
        # contract lives in the docstring and the INPUT_TYPES tooltip.
        self.assertEqual(
            _build_civitai_filename("portrait", 99, 0, 1, append_counter=False),
            "portrait.png",
        )

    def test_counter_off_batch_uses_two_digit_frame_index(self):
        # Without an in-batch suffix, N frames would all write to the same
        # path and only the last would survive. Two-digit zero-pad keeps
        # the names visibly grouped but compact.
        names = [
            _build_civitai_filename("scene", 0, i, 4, append_counter=False)
            for i in range(4)
        ]
        self.assertEqual(names, ["scene_00.png", "scene_01.png",
                                  "scene_02.png", "scene_03.png"])

    def test_counter_off_batch_size_one_drops_index_even_with_idx_zero(self):
        # batch_size==1 short-circuits the per-frame suffix path so the
        # filename stays clean. Defensive: idx and counter values shouldn't
        # leak into the result when the flag is off and batch is single.
        self.assertEqual(
            _build_civitai_filename("solo", 500, 0, 1, append_counter=False),
            "solo.png",
        )


class DateTokenExpansionTests(unittest.TestCase):
    """%date:<format>% expansion mirrors the save-image-extended-comfyui
    convention so a filename_prefix the user paste-importing from another
    pack lands without surprises. The helper is pinned at a fixed
    `now` so the assertions don't drift with the wall clock."""

    @classmethod
    def setUpClass(cls):
        # Wednesday, 2026-05-13, 19:30:45
        cls.now = time.struct_time((2026, 5, 13, 19, 30, 45, 2, 133, 0))

    def test_no_token_passes_through(self):
        self.assertEqual(
            _expand_filename_tokens("Comic_static_name", now=self.now),
            "Comic_static_name",
        )

    def test_basic_date_token(self):
        self.assertEqual(
            _expand_filename_tokens("Comic_%date:yyyy-MM-dd%", now=self.now),
            "Comic_2026-05-13",
        )

    def test_compact_datetime(self):
        self.assertEqual(
            _expand_filename_tokens("Comic_%date:yyyyMMdd_HHmmss%", now=self.now),
            "Comic_20260513_193045",
        )

    def test_split_date_and_time_underscores(self):
        self.assertEqual(
            _expand_filename_tokens("Comic_%date:yyyy-MM-dd_HH-mm%", now=self.now),
            "Comic_2026-05-13_19-30",
        )

    def test_long_month_and_weekday(self):
        self.assertEqual(
            _expand_filename_tokens("Comic_%date:MMMM_dd_yyyy%", now=self.now),
            "Comic_May_13_2026",
        )
        out = _expand_filename_tokens("Comic_%date:dddd-yyyy-MMM%", now=self.now)
        # Locale-sensitive (weekday/month abbreviations) — verify shape.
        self.assertTrue(out.startswith("Comic_") and "2026" in out)
        self.assertEqual(len(out.split("-")), 3)

    def test_12_hour_and_ampm(self):
        out = _expand_filename_tokens("Comic_%date:hh-mm_tt%", now=self.now)
        # 19:30 → 07:30 PM (locale-dependent on AM/PM literal; just check
        # the hour formatted correctly and one of AM/PM is present).
        self.assertIn("07-30_", out)
        self.assertTrue(out.endswith("PM") or out.endswith("AM"))

    def test_multiple_tokens_in_one_prefix(self):
        # Mix two %date:...% groups in the same string.
        self.assertEqual(
            _expand_filename_tokens(
                "Comic_%date:yyyy%_run_%date:HHmm%", now=self.now),
            "Comic_2026_run_1930",
        )

    def test_unknown_format_falls_through(self):
        # An unparseable format inside %date:...% stays literal so the
        # user notices on disk rather than seeing garbage timestamps.
        out = _expand_filename_tokens("Comic_%date:%not_a_format!%", now=self.now)
        # The empty fmt segment between %date: and % gets expanded with
        # nothing recognisable — strftime("") returns "". So the result
        # is "Comic_not_a_format!%". This is fine — visible artifact
        # the user can see and correct.
        self.assertIn("not_a_format", out)

    def test_does_not_touch_core_tokens(self):
        # %year% / %month% / %day% are ComfyUI's core tokens, expanded
        # downstream by folder_paths.get_save_image_path. We must leave
        # them alone — if we replaced them too the downstream pass would
        # find nothing to do.
        self.assertEqual(
            _expand_filename_tokens("Comic_%year%-%month%-%day%", now=self.now),
            "Comic_%year%-%month%-%day%",
        )

    def test_token_order_matters_yyyy_before_yy(self):
        # If we'd processed "yy" before "yyyy" the year would expand
        # twice and end up as "20262026". Pin the order is correct.
        self.assertEqual(
            _expand_filename_tokens("y%date:yyyy%y", now=self.now),
            "y2026y",
        )

    def test_token_order_MM_vs_mm(self):
        # Case-sensitive: MM=month, mm=minute. Both at once.
        self.assertEqual(
            _expand_filename_tokens("%date:MM_mm%", now=self.now),
            "05_30",
        )


class AppendCounterCoercionTests(unittest.TestCase):
    """A workflow saved in a shifted widget order can land non-bool
    values in the append_counter slot. The coercion policy is
    'default-on' — only an explicit False-like value disables the
    counter, because a silently-overwriting save is much worse UX
    than an extra suffix."""

    def test_real_true_passes_through(self):
        self.assertTrue(_coerce_append_counter(True))

    def test_real_false_passes_through(self):
        self.assertFalse(_coerce_append_counter(False))

    def test_empty_string_defaults_true(self):
        # The exact failure mode from the user's corrupted workflow:
        # widget shifted and append_counter landed on '' (the empty
        # string default of a STRING widget). Previously this read
        # as falsy → no counter → silent overwrite. Now: default True.
        self.assertTrue(_coerce_append_counter(""))

    def test_none_defaults_true(self):
        self.assertTrue(_coerce_append_counter(None))

    def test_string_false_disables_counter(self):
        # Defensive: literal "False" / "false" / "no" / "0" / "off" all
        # explicitly disable. User who deliberately serialised a string
        # value can still get the no-counter behaviour.
        for s in ("False", "false", "FALSE", "no", "0", "off", " n "):
            self.assertFalse(_coerce_append_counter(s), msg=f"{s!r} should disable")

    def test_string_true_enables_counter(self):
        for s in ("True", "true", "yes", "1", "on", " y "):
            self.assertTrue(_coerce_append_counter(s), msg=f"{s!r} should enable")

    def test_numeric_zero_disables(self):
        self.assertFalse(_coerce_append_counter(0))
        self.assertFalse(_coerce_append_counter(0.0))

    def test_numeric_nonzero_enables(self):
        self.assertTrue(_coerce_append_counter(1))
        self.assertTrue(_coerce_append_counter(-1))
        self.assertTrue(_coerce_append_counter(0.5))

    def test_unrecognised_string_defaults_true(self):
        # 'fixed' (the seed-control value that landed in append_counter
        # in the user's corrupted workflow) is unrecognised — default
        # to True so the counter still appends.
        self.assertTrue(_coerce_append_counter("fixed"))
        self.assertTrue(_coerce_append_counter("randomize"))
        self.assertTrue(_coerce_append_counter("whatever"))


class PromptIdTokenTests(unittest.TestCase):
    """Cover %prompt_id% expansion + the helpers it leans on."""

    def test_sanitize_strips_unsafe_chars(self):
        self.assertEqual(_sanitize_for_filename("cult_card_ember_initiate"),
                         "cult_card_ember_initiate")
        self.assertEqual(_sanitize_for_filename("a,b,c"), "a_b_c")
        self.assertEqual(_sanitize_for_filename("path/with spaces!"),
                         "path_with_spaces")
        self.assertEqual(_sanitize_for_filename(""), "prompt")
        self.assertEqual(_sanitize_for_filename("---"), "prompt")

    def test_extract_from_single_library_node(self):
        prompt = {
            "4": {"class_type": "PromptLibrary",
                  "inputs": {"prompt_id": "cult_card_ember_initiate"}},
        }
        self.assertEqual(_extract_prompt_library_ids(prompt),
                         ["cult_card_ember_initiate"])

    def test_extract_from_style_node(self):
        prompt = {
            "10": {"class_type": "PromptLibraryStyle",
                   "inputs": {"prompt_id": "gremmy_cartoon"}},
        }
        self.assertEqual(_extract_prompt_library_ids(prompt), ["gremmy_cartoon"])

    def test_extract_from_multi_library_3_panels(self):
        prompt = {
            "5": {"class_type": "PromptLibraryMulti", "inputs": {
                "prompt_id_1": "char_a",
                "prompt_id_2": "scene_b",
                "prompt_id_3": "bg_c",
            }},
        }
        self.assertEqual(_extract_prompt_library_ids(prompt),
                         ["char_a", "scene_b", "bg_c"])

    def test_extract_skips_save_node(self):
        # PromptLibrarySave's prompt_id input names the entry being WRITTEN,
        # not the one being used — don't pull it into the filename.
        prompt = {
            "9": {"class_type": "PromptLibrarySave",
                  "inputs": {"prompt_id": "should_be_ignored"}},
        }
        self.assertEqual(_extract_prompt_library_ids(prompt), [])

    def test_extract_skips_unrelated_nodes(self):
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "x.safetensors"}},
            "2": {"class_type": "KSampler",
                  "inputs": {"seed": 666}},
        }
        self.assertEqual(_extract_prompt_library_ids(prompt), [])

    def test_extract_handles_empty_or_missing_inputs(self):
        # An unused library node (prompt_id left blank) shouldn't show up.
        prompt = {
            "4": {"class_type": "PromptLibrary",
                  "inputs": {"prompt_id": ""}},
            "5": {"class_type": "PromptLibrary",
                  "inputs": {"prompt_id": "   "}},
            "6": {"class_type": "PromptLibrary", "inputs": {}},
            "7": {"class_type": "PromptLibrary"},
        }
        self.assertEqual(_extract_prompt_library_ids(prompt), [])

    def test_expand_single_id(self):
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "cult_card_frog"}}}
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt),
            "cult_card_frog",
        )

    def test_expand_multi_pick_comma_to_underscore(self):
        # A single PromptLibrary node can hold a comma-separated multi-pick.
        # Sanitization must collapse the commas to underscores so the
        # filename stays safe across platforms.
        prompt = {"4": {"class_type": "PromptLibrary", "inputs": {
            "prompt_id": "cult_card_ribbity_rabbity,judy_hopps_sdxl"}}}
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt),
            "cult_card_ribbity_rabbity_judy_hopps_sdxl",
        )

    def test_expand_multi_library_panels_joined(self):
        prompt = {"5": {"class_type": "PromptLibraryMulti", "inputs": {
            "prompt_id_1": "alpha", "prompt_id_2": "beta", "prompt_id_3": "gamma"}}}
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt),
            "alpha_beta_gamma",
        )

    def test_expand_fallback_when_no_loader(self):
        # No PromptLibrary loader in the workflow → never leak the literal
        # `%prompt_id%` to disk; fall back to 'GrimmRibbity'.
        prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt),
            "GrimmRibbity",
        )

    def test_expand_no_token_passes_through(self):
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "x"}}}
        self.assertEqual(
            _expand_prompt_tokens("StaticName", prompt),
            "StaticName",
        )

    def test_expand_with_none_prompt(self):
        # No workflow context at all — still fall back rather than crash.
        self.assertEqual(_expand_prompt_tokens("%prompt_id%", None),
                         "GrimmRibbity")

    def test_expand_inside_a_larger_prefix(self):
        # Token can be embedded anywhere in the prefix string.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "cult_card_frog"}}}
        self.assertEqual(
            _expand_prompt_tokens("renders/%prompt_id%_v2", prompt),
            "renders/cult_card_frog_v2",
        )


# Fixture: 5 library entries spanning the tags we use in the Random tests.
_REPLAY_ITEMS = [
    {"id": "cult_card_ember_initiate", "tags": ["Cards", "creature"]},
    {"id": "cult_card_frog",           "tags": ["Cards", "creature"]},
    {"id": "cult_card_drown_kid",      "tags": ["Cards", "creature"]},
    {"id": "cult_card_beast_pit",      "tags": ["Cards", "trap"]},
    {"id": "gremmy_cartoon",           "tags": ["gremmy"]},
]


class ReplayRandomPickTests(unittest.TestCase):
    """The pure-function half of the Random replay path."""

    def test_seed_picks_deterministically_within_filter(self):
        # Same (seed, tag_filter, items) must always produce the same id —
        # this is the only reason we can replay the pick at save time
        # without wiring anything from the Random node.
        a = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 42)
        b = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 42)
        self.assertEqual(a, b)
        # And the picked id must actually be Cards-tagged.
        picked = next(i for i in _REPLAY_ITEMS if i["id"] == a)
        self.assertIn("Cards", picked["tags"])

    def test_different_seeds_can_diverge(self):
        # Sanity: across a reasonable seed sweep we see more than one
        # distinct pick (otherwise the test above is vacuous).
        seeds = range(0, 50)
        ids = {_replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", s)
               for s in seeds}
        self.assertGreater(len(ids), 1)

    def test_tag_filter_and_aware(self):
        # Multi-tag filter is AND: every tag must be on the entry. Only
        # the trap card has both Cards+trap, so the pick is forced.
        pid = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards,trap", 0)
        self.assertEqual(pid, "cult_card_beast_pit")

    def test_empty_filter_matches_all(self):
        pid = _replay_random_pick_from_items(_REPLAY_ITEMS, "", 0)
        self.assertIn(pid, [i["id"] for i in _REPLAY_ITEMS])

    def test_no_matches_returns_empty(self):
        self.assertEqual(
            _replay_random_pick_from_items(_REPLAY_ITEMS, "no_such_tag", 0),
            "",
        )

    def test_non_list_items_returns_empty(self):
        # Defensive — bad return from the lazy fetcher.
        self.assertEqual(_replay_random_pick_from_items(None, "", 0), "")
        self.assertEqual(_replay_random_pick_from_items({}, "", 0), "")

    def test_bad_seed_returns_empty(self):
        self.assertEqual(
            _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", "not-an-int"),
            "",
        )

    def test_missing_seed_defaults_to_zero(self):
        # None seed → 0. Still deterministic, never crashes.
        pid = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", None)
        self.assertEqual(pid,
                         _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 0))


class RandomPickWorkflowTests(unittest.TestCase):
    """Walk a real-shaped workflow dict + assert the right ids surface
    in %prompt_id% expansion. items_fetcher is injected so the test
    doesn't touch the real prompts.json on disk."""

    def _fetcher(self):
        return list(_REPLAY_ITEMS)

    def test_random_node_replay_produces_picked_id(self):
        # The exact case the user wants to work without wiring: workflow
        # has a PromptLibraryRandom and a sampler; no PromptLibrary
        # loader. The save node walks, replays, and names the file.
        prompt = {
            "5": {"class_type": "PromptLibraryRandom",
                  "inputs": {"tag_filter": "Cards", "seed": 42}},
            "9": {"class_type": "KSampler", "inputs": {"seed": 666}},
        }
        expected_id = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 42)
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt,
                                   items_fetcher=self._fetcher),
            expected_id,
        )

    def test_extract_random_pick_ids_skips_when_no_random_node(self):
        # If the workflow has no Random node we never call the fetcher
        # — and the result is empty.
        called = []

        def fetcher():
            called.append(1)
            return list(_REPLAY_ITEMS)

        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "x"}}}
        self.assertEqual(_extract_random_pick_ids(prompt, items_fetcher=fetcher), [])
        self.assertFalse(called, "fetcher should not be called without a Random node")

    def test_static_loader_and_random_combine(self):
        # Mixed graph: a static PromptLibrary picks character_a, and a
        # Random by Tag picks the scene from the Cards pool. Both
        # collapse into one underscore-joined filename token, in walk
        # order (loaders first, then randoms).
        prompt = {
            "4": {"class_type": "PromptLibrary",
                  "inputs": {"prompt_id": "character_a"}},
            "5": {"class_type": "PromptLibraryRandom",
                  "inputs": {"tag_filter": "Cards", "seed": 42}},
        }
        random_pick = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 42)
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt,
                                   items_fetcher=self._fetcher),
            f"character_a_{random_pick}",
        )

    def test_multiple_random_nodes_join_in_walk_order(self):
        # Overnight loop with two Random panels (e.g. character + scene).
        prompt = {
            "5": {"class_type": "PromptLibraryRandom",
                  "inputs": {"tag_filter": "Cards", "seed": 1}},
            "6": {"class_type": "PromptLibraryRandom",
                  "inputs": {"tag_filter": "Cards", "seed": 2}},
        }
        a = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 1)
        b = _replay_random_pick_from_items(_REPLAY_ITEMS, "Cards", 2)
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt,
                                   items_fetcher=self._fetcher),
            f"{a}_{b}",
        )

    def test_random_node_with_no_matches_falls_back(self):
        # Random node's tag_filter doesn't match any library entry — the
        # replay returns "", we drop empties, and nothing else feeds the
        # token → fallback to 'GrimmRibbity'.
        prompt = {
            "5": {"class_type": "PromptLibraryRandom",
                  "inputs": {"tag_filter": "no_such_tag", "seed": 0}},
        }
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt,
                                   items_fetcher=self._fetcher),
            "GrimmRibbity",
        )

    def test_fetcher_returning_empty_falls_back(self):
        # The package wasn't importable (test harness, standalone) so
        # the lazy fetcher returns []. Random replay produces nothing,
        # so the static walk's result (or fallback) wins.
        prompt = {
            "5": {"class_type": "PromptLibraryRandom",
                  "inputs": {"tag_filter": "Cards", "seed": 0}},
        }
        self.assertEqual(
            _expand_prompt_tokens("%prompt_id%", prompt,
                                   items_fetcher=lambda: []),
            "GrimmRibbity",
        )


# Realistic items list reused across the snapshot tests — mirrors the
# shape we actually ship: id/name/tags/negative present, loras a mix of
# old (lora_name/strength) and new (name/strength_model/strength_clip)
# shapes so the normalizer is exercised end-to-end.
_SNAPSHOT_ITEMS = [
    {
        "id": "cult_card_ghoul",
        "name": "Pit Ghoul -- Cult Card",
        "tags": ["Cards", "Cards:nuke", "Cards:creature"],
        "negative": "lowres, blurry, watermark",
        "loras": [
            {"name": "wasteland.safetensors", "strength_model": 0.7,
             "strength_clip": 0.6, "enabled": True},
        ],
    },
    {
        "id": "gremmy_cartoon",
        "name": "Cartoon Gremmy",
        "tags": ["Gremmy"],
        "negative": "noisy",
        "loras": [
            {"lora_name": "gremmy_v1.safetensors", "strength": 1.0},
        ],
    },
    {
        "id": "raw_neg",
        "name": "Raw",
        "tags": ["Cards"],
        "negative": "x" * 5000,
        "loras": [],
    },
]


class LoraNormalizerTests(unittest.TestCase):
    """Round-trip the per-LoRA shape coercion the snapshot depends on."""

    def test_new_shape_passthrough(self):
        out = _normalize_lora_for_snapshot({
            "name": "x.safetensors", "strength_model": 0.7,
            "strength_clip": 0.6, "enabled": True,
        })
        self.assertEqual(out["name"], "x.safetensors")
        self.assertAlmostEqual(out["strength_model"], 0.7)
        self.assertAlmostEqual(out["strength_clip"], 0.6)
        self.assertTrue(out["enabled"])

    def test_legacy_shape_normalized(self):
        # Old library entries used lora_name + a single strength field.
        out = _normalize_lora_for_snapshot({
            "lora_name": "old.safetensors", "strength": 0.8,
        })
        self.assertEqual(out["name"], "old.safetensors")
        self.assertAlmostEqual(out["strength_model"], 0.8)
        self.assertAlmostEqual(out["strength_clip"], 0.8)
        self.assertTrue(out["enabled"])

    def test_missing_name_returns_none(self):
        self.assertIsNone(_normalize_lora_for_snapshot({"strength": 1.0}))
        self.assertIsNone(_normalize_lora_for_snapshot({}))
        self.assertIsNone(_normalize_lora_for_snapshot(None))
        self.assertIsNone(_normalize_lora_for_snapshot("not a dict"))

    def test_bad_strength_defaults_to_one(self):
        out = _normalize_lora_for_snapshot({
            "name": "x.safetensors", "strength_model": "not-a-number",
        })
        self.assertAlmostEqual(out["strength_model"], 1.0)


class ResolveIndividualPidsTests(unittest.TestCase):
    """The pid-set the snapshot iterates over: multi-pick gets split."""

    def test_single_pid_passthrough(self):
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "cult_card_ghoul"}}}
        self.assertEqual(
            _resolve_individual_pids(prompt, items_fetcher=lambda: []),
            ["cult_card_ghoul"],
        )

    def test_multi_pick_split(self):
        # Filename naming joins "a,b" → "a_b" — snapshot SPLITS to ["a","b"]
        # so each entry gets serialized once.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "a, b ,c"}}}
        self.assertEqual(
            _resolve_individual_pids(prompt, items_fetcher=lambda: []),
            ["a", "b", "c"],
        )

    def test_dedup_across_loaders(self):
        # If two PromptLibrary nodes happen to reference the same id, the
        # snapshot should still list it once.
        prompt = {
            "4": {"class_type": "PromptLibrary",
                  "inputs": {"prompt_id": "shared"}},
            "5": {"class_type": "PromptLibraryStyle",
                  "inputs": {"prompt_id": "shared"}},
        }
        self.assertEqual(
            _resolve_individual_pids(prompt, items_fetcher=lambda: []),
            ["shared"],
        )


class LibrarySnapshotTests(unittest.TestCase):
    """End-to-end shape of the grimmribbity_library PNG chunk payload."""

    def _fetcher(self):
        return list(_SNAPSHOT_ITEMS)

    def test_no_loader_returns_none(self):
        # No library node in the workflow → snapshot is None and the save
        # path skips writing the chunk entirely. Keeps non-library saves
        # clean.
        prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        self.assertIsNone(
            _build_library_entry_snapshot(prompt, items_fetcher=self._fetcher))

    def test_single_entry_shape(self):
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "cult_card_ghoul"}}}
        snap = _build_library_entry_snapshot(prompt, items_fetcher=self._fetcher)
        self.assertEqual(snap["schema"], 1)
        self.assertEqual(snap["source"], "GrimmRibbity")
        self.assertEqual(len(snap["entries"]), 1)
        e = snap["entries"][0]
        self.assertEqual(e["id"], "cult_card_ghoul")
        self.assertEqual(e["name"], "Pit Ghoul -- Cult Card")
        self.assertEqual(e["tags"], ["Cards", "Cards:nuke", "Cards:creature"])
        self.assertEqual(len(e["loras"]), 1)
        self.assertEqual(e["loras"][0]["name"], "wasteland.safetensors")
        self.assertAlmostEqual(e["loras"][0]["strength_clip"], 0.6)

    def test_multi_pick_expands_into_separate_entries(self):
        # Comma-split in one library node = N entries in the snapshot.
        # Distinct from the filename builder which joins with underscores.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "cult_card_ghoul,gremmy_cartoon"}}}
        snap = _build_library_entry_snapshot(prompt, items_fetcher=self._fetcher)
        ids = [e["id"] for e in snap["entries"]]
        self.assertEqual(ids, ["cult_card_ghoul", "gremmy_cartoon"])

    def test_missing_entry_flagged_not_skipped(self):
        # An id that no longer exists in the library still gets a stub —
        # tells the consumer "this was referenced but is gone" rather than
        # dropping silently.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "ghost_entry"}}}
        snap = _build_library_entry_snapshot(prompt, items_fetcher=self._fetcher)
        self.assertEqual(len(snap["entries"]), 1)
        e = snap["entries"][0]
        self.assertEqual(e["id"], "ghost_entry")
        self.assertTrue(e.get("missing"))
        # No name/tags/loras keys on the stub — consumer must check "missing".
        self.assertNotIn("name", e)

    def test_negative_truncated_with_ellipsis(self):
        # Negatives can run 700+ chars on cult cards. The snapshot caps at
        # _SNAPSHOT_NEGATIVE_MAX (default 500) + "..." so the PNG chunk
        # stays compact across a 100-frame batch save.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "raw_neg"}}}
        snap = _build_library_entry_snapshot(prompt, items_fetcher=self._fetcher,
                                              negative_max=80)
        neg = snap["entries"][0]["negative"]
        self.assertLessEqual(len(neg), 83)  # 80 chars + "..."
        self.assertTrue(neg.endswith("..."))

    def test_random_node_replay_resolves(self):
        # PromptLibraryRandom doesn't expose its picked id as an input —
        # the snapshot reuses the same deterministic replay path the
        # filename builder uses. The replayed id then resolves to the
        # full entry record.
        prompt = {"5": {"class_type": "PromptLibraryRandom",
                        "inputs": {"tag_filter": "Cards:nuke", "seed": 0}}}
        items = [
            {"id": "cult_card_ghoul", "tags": ["Cards:nuke"],
             "negative": "x", "loras": []},
        ]
        snap = _build_library_entry_snapshot(prompt, items_fetcher=lambda: items)
        self.assertEqual(snap["entries"][0]["id"], "cult_card_ghoul")

    def test_fetcher_returning_empty_returns_none(self):
        # Standalone install / test harness: lazy fetcher returns []. No
        # entries resolve, so the snapshot returns None and the save path
        # doesn't write a chunk for an unresolvable workflow.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "x"}}}
        snap = _build_library_entry_snapshot(prompt, items_fetcher=lambda: [])
        # x has no record, but it's still mentioned → snapshot keeps the
        # stub. (None only when there's nothing to mention at all.)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["entries"][0]["id"], "x")
        self.assertTrue(snap["entries"][0].get("missing"))

    def test_snapshot_serializes_to_json(self):
        # The save loop json.dumps the snapshot before writing the PNG
        # chunk. Defensive check: nothing in the snapshot is non-JSON.
        prompt = {"4": {"class_type": "PromptLibrary",
                        "inputs": {"prompt_id": "cult_card_ghoul,gremmy_cartoon"}}}
        snap = _build_library_entry_snapshot(prompt, items_fetcher=self._fetcher)
        encoded = json.dumps(snap, ensure_ascii=False)
        # And it round-trips.
        self.assertEqual(json.loads(encoded), snap)


class LibrarySnapshotPngRoundTripTests(unittest.TestCase):
    """Confirm the PIL chunk-write path used by save() actually carries
    the JSON through a real PNG encode/decode cycle. Doesn't call save()
    (which needs torch + folder_paths) — exercises the PNG layer."""

    def test_chunk_writes_and_reads_back(self):
        from PIL import Image, PngImagePlugin
        import io
        snap = {"schema": 1, "source": "GrimmRibbity",
                "entries": [{"id": "cult_card_frog", "tags": ["Cards"]}]}
        info = PngImagePlugin.PngInfo()
        info.add_text("grimmribbity_library", json.dumps(snap))
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (0, 0, 0)).save(
            buf, format="PNG", pnginfo=info)
        buf.seek(0)
        reloaded = Image.open(buf)
        # PIL surfaces text chunks as .info[key] after load().
        reloaded.load()
        self.assertIn("grimmribbity_library", reloaded.info)
        self.assertEqual(
            json.loads(reloaded.info["grimmribbity_library"]),
            snap,
        )


if __name__ == "__main__":
    unittest.main()
