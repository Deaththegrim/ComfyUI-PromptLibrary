import json
import unittest

from civitai_save import build_a1111_parameters, _build_civitai_filename


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


if __name__ == "__main__":
    unittest.main()
