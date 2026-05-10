"""Unit tests for GrimmRibbity Upscale SDXL.

Real comfy / nodes / torch are not guaranteed in the bare test env, so
this file stubs the comfy-side modules in sys.modules before importing
upscaler_node. We test the pure-Python tile arithmetic + feather-mask
helpers + INPUT_TYPES shape; full sampler integration is verified in
real ComfyUI, not via mocks.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


def _install_stubs():
    if "torch" not in sys.modules:
        return False
    fake_nodes = types.ModuleType("nodes")
    fake_nodes.common_ksampler = lambda *a, **k: ({"samples": None},)

    class _Apply:
        def apply_controlnet(self, **k):
            return (k["positive"], k["negative"])

    class _Upscale:
        def upscale(self, _model, image):
            return (image,)

    fake_nodes.ControlNetApplyAdvanced = _Apply
    fake_nodes.ImageUpscaleWithModel = _Upscale
    sys.modules["nodes"] = fake_nodes

    fake_samplers = types.ModuleType("comfy.samplers")

    class _KSampler:
        SAMPLERS = ["euler"]
        SCHEDULERS = ["simple"]

    fake_samplers.KSampler = _KSampler
    sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sys.modules["comfy.samplers"] = fake_samplers
    sys.modules["comfy"].samplers = fake_samplers

    fake_utils = types.ModuleType("comfy.utils")

    class _PBar:
        def __init__(self, _n):
            pass

        def update(self, _n):
            pass

    fake_utils.ProgressBar = _PBar
    fake_utils.common_upscale = lambda samples, w, h, *_a, **_k: samples
    sys.modules["comfy.utils"] = fake_utils
    sys.modules["comfy"].utils = fake_utils
    return True


import importlib.util
_TORCH_OK = importlib.util.find_spec("torch") is not None


if _TORCH_OK and _install_stubs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import upscaler_node  # noqa: E402
else:
    upscaler_node = None


@unittest.skipUnless(upscaler_node is not None, "torch / comfy stubs not available")
class TileGridTests(unittest.TestCase):
    def test_single_tile_when_image_smaller_than_tile(self):
        tiles = upscaler_node._compute_tile_grid(400, 400, 512, 512, 32, True)
        self.assertEqual(len(tiles), 1)
        t = tiles[0]
        self.assertEqual((t.inner_x, t.inner_y, t.inner_w, t.inner_h), (0, 0, 400, 400))

    def test_grid_count_matches_ceiling_division(self):
        tiles = upscaler_node._compute_tile_grid(1024, 1024, 512, 512, 32, False)
        self.assertEqual(len(tiles), 4)

    def test_inner_tiles_cover_image_without_gaps(self):
        tiles = upscaler_node._compute_tile_grid(1024, 768, 512, 512, 32, False)
        covered = sum(t.inner_w * t.inner_h for t in tiles)
        self.assertEqual(covered, 1024 * 768)

    def test_force_uniform_keeps_tile_size(self):
        tiles = upscaler_node._compute_tile_grid(1100, 1100, 512, 512, 32, True)
        for t in tiles:
            self.assertEqual(t.inner_w, 512)
            self.assertEqual(t.inner_h, 512)

    def test_force_uniform_off_allows_short_trailing_tile(self):
        tiles = upscaler_node._compute_tile_grid(1100, 1100, 512, 512, 32, False)
        widths = sorted({t.inner_w for t in tiles})
        self.assertEqual(widths, [76, 512])

    def test_outer_padding_clamps_to_image(self):
        tiles = upscaler_node._compute_tile_grid(1024, 1024, 512, 512, 32, True)
        for t in tiles:
            self.assertGreaterEqual(t.outer_x, 0)
            self.assertGreaterEqual(t.outer_y, 0)
            self.assertLessEqual(t.outer_x + t.outer_w, 1024)
            self.assertLessEqual(t.outer_y + t.outer_h, 1024)

    def test_invalid_tile_size_raises(self):
        with self.assertRaises(ValueError):
            upscaler_node._compute_tile_grid(1024, 1024, 0, 512, 32, True)


@unittest.skipUnless(upscaler_node is not None, "torch / comfy stubs not available")
class RoundToEightTests(unittest.TestCase):
    def test_already_aligned(self):
        self.assertEqual(upscaler_node._round_to_8(512), 512)

    def test_rounds_up(self):
        self.assertEqual(upscaler_node._round_to_8(513), 520)
        self.assertEqual(upscaler_node._round_to_8(519), 520)

    def test_zero(self):
        self.assertEqual(upscaler_node._round_to_8(0), 0)


@unittest.skipUnless(upscaler_node is not None, "torch / comfy stubs not available")
class FeatherMaskTests(unittest.TestCase):
    def test_mask_shape(self):
        mask = upscaler_node._make_feather_mask(64, 96, 8, "cpu")
        self.assertEqual(tuple(mask.shape), (64, 96))

    def test_center_is_one(self):
        mask = upscaler_node._make_feather_mask(64, 64, 8, "cpu")
        self.assertAlmostEqual(float(mask[32, 32]), 1.0, places=5)

    def test_corner_falls_off(self):
        mask = upscaler_node._make_feather_mask(64, 64, 8, "cpu")
        self.assertLess(float(mask[0, 0]), 0.5)

    def test_blur_zero_returns_all_ones(self):
        import torch as _t
        mask = upscaler_node._make_feather_mask(16, 16, 0, "cpu")
        self.assertTrue(_t.allclose(mask, _t.ones(16, 16)))


@unittest.skipUnless(upscaler_node is not None, "torch / comfy stubs not available")
class InputTypesTests(unittest.TestCase):
    def test_required_inputs_present(self):
        spec = upscaler_node.GrimmRibbityUpscaleSDXL.INPUT_TYPES()
        for k in ("image", "model", "vae", "positive", "negative",
                  "upscale_by", "denoise", "steps", "cfg",
                  "sampler_name", "scheduler", "seed",
                  "tile_width", "tile_height", "tile_padding", "mask_blur",
                  "force_uniform_tiles", "cn_strength",
                  "seam_fix_mode", "bypass"):
            self.assertIn(k, spec["required"], f"missing required input {k}")

    def test_optional_inputs_are_optional(self):
        spec = upscaler_node.GrimmRibbityUpscaleSDXL.INPUT_TYPES()
        self.assertIn("control_net", spec["optional"])
        self.assertIn("upscale_model", spec["optional"])

    def test_defaults_match_clarity_recipe(self):
        spec = upscaler_node.GrimmRibbityUpscaleSDXL.INPUT_TYPES()
        self.assertEqual(spec["required"]["denoise"][1]["default"], 0.25)
        self.assertEqual(spec["required"]["cfg"][1]["default"], 4.0)
        self.assertEqual(spec["required"]["upscale_by"][1]["default"], 2.0)
        self.assertEqual(spec["required"]["cn_strength"][1]["default"], 0.5)
        self.assertEqual(spec["required"]["tile_width"][1]["default"], 512)

    def test_return_types(self):
        self.assertEqual(upscaler_node.GrimmRibbityUpscaleSDXL.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(upscaler_node.GrimmRibbityUpscaleSDXL.CATEGORY, "GrimmRibbity/Upscale")


if __name__ == "__main__":
    unittest.main()
