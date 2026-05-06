"""Unit tests for GrimmRibbity Comic Page (Regional).

After the self-contained refactor (no Inspire Pack dependency), the node
calls ComfyUI core's CLIPTextEncode + ConditioningSetMask via the runtime
nodes.NODE_CLASS_MAPPINGS registry, plus its own _color_to_mask /
_hex_to_rgb01 helpers. Tests mock the registry-resolved core nodes and
exercise the helpers directly with real torch tensors.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


def _install_fake_nodes_module():
    """Inject a fake `nodes` module whose NODE_CLASS_MAPPINGS contains
    fakes for the two ComfyUI core nodes the wrapper calls. Returns the
    encoder + masker call logs."""
    encode_calls: list = []
    mask_calls: list = []

    class FakeCLIPTextEncode:
        def encode(self, clip, prompt):
            encode_calls.append({"clip": clip, "prompt": prompt})
            # CLIPTextEncode returns (CONDITIONING,) — a 1-tuple where
            # CONDITIONING is a list of (tensor, dict) tuples. Mimic shape.
            return ([(f"enc({prompt[:20]})", {})],)

    class FakeConditioningSetMask:
        def append(self, conditioning, mask, set_cond_area, strength):
            mask_calls.append({
                "conditioning": conditioning, "mask": mask,
                "set_cond_area": set_cond_area, "strength": strength,
            })
            # ConditioningSetMask returns (CONDITIONING,) — same shape.
            tagged = [(f"masked({c[0]})", {**c[1], "mask_set": True})
                      for c in conditioning]
            return (tagged,)

    fake_nodes = types.ModuleType("nodes")
    fake_nodes.NODE_CLASS_MAPPINGS = {
        "CLIPTextEncode": FakeCLIPTextEncode,
        "ConditioningSetMask": FakeConditioningSetMask,
    }
    sys.modules["nodes"] = fake_nodes
    return encode_calls, mask_calls


_ENCODE_CALLS, _MASK_CALLS = _install_fake_nodes_module()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comic_page  # noqa: E402

# torch is optional in the headless test venv (.testenv runs without it).
# Tests that need real tensors are skipped when it's missing — the hex
# parser tests still run because they're pure-Python.
try:
    import torch  # noqa: E402
except ImportError:
    torch = None
_NEEDS_TORCH = unittest.skipIf(torch is None, "torch not installed in this venv")


class HexParseTests(unittest.TestCase):
    def test_parses_pure_red(self):
        self.assertEqual(comic_page._hex_to_rgb01("#FF0000"), (1.0, 0.0, 0.0))

    def test_parses_lowercase_no_hash(self):
        self.assertEqual(comic_page._hex_to_rgb01("00ff00"), (0.0, 1.0, 0.0))

    def test_parses_mixed_value(self):
        r, g, b = comic_page._hex_to_rgb01("#80c0ff")
        self.assertAlmostEqual(r, 128/255, places=4)
        self.assertAlmostEqual(g, 192/255, places=4)
        self.assertAlmostEqual(b, 1.0,     places=4)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            comic_page._hex_to_rgb01("#FFF")  # 3-char shorthand not supported


@_NEEDS_TORCH
class ColorToMaskTests(unittest.TestCase):
    def test_extracts_red_region_only(self):
        # 4x4 image: top-left red, top-right green, bottom-left blue, bottom-right yellow.
        img = torch.zeros(1, 4, 4, 3)
        img[0, 0:2, 0:2] = torch.tensor([1.0, 0.0, 0.0])    # red
        img[0, 0:2, 2:4] = torch.tensor([0.0, 1.0, 0.0])    # green
        img[0, 2:4, 0:2] = torch.tensor([0.0, 0.0, 1.0])    # blue
        img[0, 2:4, 2:4] = torch.tensor([1.0, 1.0, 0.0])    # yellow

        red_mask = comic_page._color_to_mask(img, "#FF0000")
        # Red occupies rows 0-1 cols 0-1 — 4 pixels.
        self.assertEqual(red_mask.shape, (1, 4, 4))
        self.assertEqual(red_mask.sum().item(), 4.0)
        self.assertEqual(red_mask[0, 0, 0].item(), 1.0)
        self.assertEqual(red_mask[0, 0, 2].item(), 0.0)  # green region

    def test_no_match_yields_empty_mask(self):
        img = torch.zeros(1, 2, 2, 3)
        img[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        # Looking for a color that isn't in the image.
        mask = comic_page._color_to_mask(img, "#0000FF")
        self.assertEqual(mask.sum().item(), 0.0)

    def test_tolerance_swallows_anti_alias_jitter(self):
        # Pixel that's "almost red" (R=0.99 instead of 1.0). Should still match.
        img = torch.zeros(1, 1, 1, 3)
        img[0, 0, 0] = torch.tensor([0.99, 0.005, 0.005])
        mask = comic_page._color_to_mask(img, "#FF0000")
        self.assertEqual(mask.sum().item(), 1.0)


class ComicPageInputShapeTests(unittest.TestCase):
    """Pure-schema tests — no torch needed."""

    def test_input_types_exposes_six_panel_widgets(self):
        spec = comic_page.GrimmRibbityComicPage.INPUT_TYPES()
        for i in range(1, 7):
            self.assertIn(f"panel_{i}_prompt", spec["required"])
            self.assertIn(f"panel_{i}_color", spec["required"])

    def test_default_panel_colors_cycle_red_green_blue_yellow(self):
        spec = comic_page.GrimmRibbityComicPage.INPUT_TYPES()
        self.assertEqual(spec["required"]["panel_1_color"][1]["default"], "#FF0000")
        self.assertEqual(spec["required"]["panel_2_color"][1]["default"], "#00FF00")
        self.assertEqual(spec["required"]["panel_3_color"][1]["default"], "#0000FF")
        self.assertEqual(spec["required"]["panel_4_color"][1]["default"], "#FFFF00")


@_NEEDS_TORCH
class ComicPageNodeTests(unittest.TestCase):
    def setUp(self):
        _ENCODE_CALLS.clear()
        _MASK_CALLS.clear()
        self.node = comic_page.GrimmRibbityComicPage()
        # 2x2 image with one pixel of each default color, large enough that
        # the build() doesn't trip on degenerate masks.
        img = torch.zeros(1, 2, 2, 3)
        img[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])  # red
        img[0, 0, 1] = torch.tensor([0.0, 1.0, 0.0])  # green
        img[0, 1, 0] = torch.tensor([0.0, 0.0, 1.0])  # blue
        img[0, 1, 1] = torch.tensor([1.0, 1.0, 0.0])  # yellow
        self.layout = img

    def test_active_panels_each_invoke_clip_and_mask_once(self):
        out = self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="", strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="a knight", panel_1_color="#FF0000",
            panel_2_prompt="a wizard", panel_2_color="#00FF00",
            panel_3_prompt="",        panel_3_color="#0000FF",
            panel_4_prompt="",        panel_4_color="#FFFF00",
            panel_5_prompt="",        panel_5_color="#FF00FF",
            panel_6_prompt="",        panel_6_color="#00FFFF",
        )
        # 2 active panels → 2 encode + 2 mask calls.
        self.assertEqual(len(_ENCODE_CALLS), 2)
        self.assertEqual(len(_MASK_CALLS), 2)
        self.assertEqual(_ENCODE_CALLS[0]["prompt"], "a knight")
        self.assertEqual(_ENCODE_CALLS[1]["prompt"], "a wizard")
        # 1 entry per panel → 2 entries in the combined output.
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2)

    def test_skips_panels_with_blank_prompts(self):
        self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="", strength=1.0, set_cond_area="default",
            panel_1_prompt="",  panel_1_color="#FF0000",
            panel_2_prompt="x", panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="y", panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        self.assertEqual([c["prompt"] for c in _ENCODE_CALLS], ["x", "y"])

    def test_skips_panels_with_blank_color(self):
        self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="", strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="x", panel_1_color="",       # missing color → skip
            panel_2_prompt="y", panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="",  panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        self.assertEqual(len(_ENCODE_CALLS), 1)
        self.assertEqual(_ENCODE_CALLS[0]["prompt"], "y")

    def test_no_active_panels_returns_empty_conditioning(self):
        out = self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="", strength=1.0, set_cond_area="default",
            panel_1_prompt="", panel_1_color="#FF0000",
            panel_2_prompt="", panel_2_color="#00FF00",
            panel_3_prompt="", panel_3_color="#0000FF",
            panel_4_prompt="", panel_4_color="#FFFF00",
            panel_5_prompt="", panel_5_color="#FF00FF",
            panel_6_prompt="", panel_6_color="#00FFFF",
        )
        self.assertEqual(_ENCODE_CALLS, [])
        self.assertEqual(_MASK_CALLS, [])
        self.assertEqual(out, ([],))

    def test_strength_and_set_cond_area_propagate(self):
        self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="", strength=0.6, set_cond_area="default",
            panel_1_prompt="x", panel_1_color="#FF0000",
            panel_2_prompt="",  panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="",  panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        self.assertEqual(_MASK_CALLS[0]["strength"], 0.6)
        self.assertEqual(_MASK_CALLS[0]["set_cond_area"], "default")

    def test_shared_prompt_prepends_to_every_panel(self):
        # The consistency-fix feature: shared_prompt prefixes every non-blank
        # panel before CLIP encoding. Empty panels are still skipped.
        self.node.build(
            clip="CLIP", panel_layout=self.layout,
            shared_prompt="jessica vale, blue_hair",
            strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="standing in forest", panel_1_color="#FF0000",
            panel_2_prompt="action pose",        panel_2_color="#00FF00",
            panel_3_prompt="",                    panel_3_color="#0000FF",
            panel_4_prompt="",                    panel_4_color="#FFFF00",
            panel_5_prompt="",                    panel_5_color="#FF00FF",
            panel_6_prompt="",                    panel_6_color="#00FFFF",
        )
        self.assertEqual(len(_ENCODE_CALLS), 2)
        self.assertEqual(_ENCODE_CALLS[0]["prompt"],
                          "jessica vale, blue_hair, standing in forest")
        self.assertEqual(_ENCODE_CALLS[1]["prompt"],
                          "jessica vale, blue_hair, action pose")

    def test_blank_shared_prompt_doesnt_alter_panel_prompt(self):
        self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="",
            strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="just this", panel_1_color="#FF0000",
            panel_2_prompt="",          panel_2_color="#00FF00",
            panel_3_prompt="",          panel_3_color="#0000FF",
            panel_4_prompt="",          panel_4_color="#FFFF00",
            panel_5_prompt="",          panel_5_color="#FF00FF",
            panel_6_prompt="",          panel_6_color="#00FFFF",
        )
        self.assertEqual(_ENCODE_CALLS[0]["prompt"], "just this")

    def test_mask_passed_to_set_mask_is_correct_shape(self):
        # The mask we build via _color_to_mask should be the exact tensor
        # passed into ConditioningSetMask — verify shape preservation.
        self.node.build(
            clip="CLIP", panel_layout=self.layout, shared_prompt="", strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="x", panel_1_color="#FF0000",
            panel_2_prompt="",  panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="",  panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        mask = _MASK_CALLS[0]["mask"]
        self.assertEqual(mask.shape, (1, 2, 2))
        # Red is at (0,0). Mask should have a 1 there and 0s elsewhere.
        self.assertEqual(mask[0, 0, 0].item(), 1.0)
        self.assertEqual(mask[0, 0, 1].item(), 0.0)
        self.assertEqual(mask[0, 1, 0].item(), 0.0)
        self.assertEqual(mask[0, 1, 1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
