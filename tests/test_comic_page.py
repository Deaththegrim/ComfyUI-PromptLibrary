"""Unit tests for GrimmRibbity Comic Page (Regional).

Inspire Pack is registered via ComfyUI's global `nodes.NODE_CLASS_MAPPINGS`
at runtime. The test environment doesn't have ComfyUI's `nodes` module on
the path, so we inject a fake `nodes` module with a fake
`RegionalConditioningColorMask` to verify wiring + per-panel iteration.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


def _install_fake_nodes_module():
    """Inject a fake `nodes` module whose NODE_CLASS_MAPPINGS contains a
    fake RegionalConditioningColorMask. Returns the call log."""
    calls = []

    class FakeRegionalCondColorMask:
        def doit(self, clip, color_mask, mask_color, strength, set_cond_area, prompt):
            calls.append({
                "clip": clip, "color_mask": color_mask, "mask_color": mask_color,
                "strength": strength, "set_cond_area": set_cond_area, "prompt": prompt,
            })
            # Return a single-entry CONDITIONING — a list of one tuple — same
            # shape Inspire Pack returns.
            return ([(f"cond({mask_color}:{prompt[:20]})", {})],)

    fake_nodes = types.ModuleType("nodes")
    fake_nodes.NODE_CLASS_MAPPINGS = {
        "RegionalConditioningColorMask": FakeRegionalCondColorMask,
    }
    sys.modules["nodes"] = fake_nodes
    return calls


_CALLS = _install_fake_nodes_module()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comic_page  # noqa: E402


class ComicPageTests(unittest.TestCase):
    def setUp(self):
        _CALLS.clear()
        self.node = comic_page.GrimmRibbityComicPage()

    def test_input_types_exposes_six_panel_widgets(self):
        spec = comic_page.GrimmRibbityComicPage.INPUT_TYPES()
        for i in range(1, 7):
            self.assertIn(f"panel_{i}_prompt", spec["required"])
            self.assertIn(f"panel_{i}_color", spec["required"])
        # CLIP + layout + global controls
        self.assertIn("clip", spec["required"])
        self.assertIn("panel_layout", spec["required"])
        self.assertIn("strength", spec["required"])
        self.assertIn("set_cond_area", spec["required"])

    def test_default_panel_colors_cycle_red_green_blue_yellow(self):
        spec = comic_page.GrimmRibbityComicPage.INPUT_TYPES()
        self.assertEqual(spec["required"]["panel_1_color"][1]["default"], "#FF0000")
        self.assertEqual(spec["required"]["panel_2_color"][1]["default"], "#00FF00")
        self.assertEqual(spec["required"]["panel_3_color"][1]["default"], "#0000FF")
        self.assertEqual(spec["required"]["panel_4_color"][1]["default"], "#FFFF00")

    def test_active_panels_each_invoke_regional_conditioning(self):
        out = self.node.build(
            clip="CLIP", panel_layout="MASK", strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="a knight", panel_1_color="#FF0000",
            panel_2_prompt="a wizard", panel_2_color="#00FF00",
            panel_3_prompt="",        panel_3_color="#0000FF",
            panel_4_prompt="",        panel_4_color="#FFFF00",
            panel_5_prompt="",        panel_5_color="#FF00FF",
            panel_6_prompt="",        panel_6_color="#00FFFF",
        )
        self.assertEqual(len(_CALLS), 2, "should call regional cond once per non-empty panel")
        self.assertEqual(_CALLS[0]["mask_color"], "#FF0000")
        self.assertEqual(_CALLS[0]["prompt"], "a knight")
        self.assertEqual(_CALLS[1]["mask_color"], "#00FF00")
        self.assertEqual(_CALLS[1]["prompt"], "a wizard")
        # Output combines into one CONDITIONING list
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2, "two panel conditionings concatenated")

    def test_skips_panels_with_blank_prompts(self):
        self.node.build(
            clip="CLIP", panel_layout="MASK", strength=1.0, set_cond_area="default",
            panel_1_prompt="",  panel_1_color="#FF0000",
            panel_2_prompt="x", panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="y", panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        # Only panels 2 and 4 had prompts.
        self.assertEqual(len(_CALLS), 2)
        self.assertEqual(_CALLS[0]["prompt"], "x")
        self.assertEqual(_CALLS[1]["prompt"], "y")

    def test_skips_panels_with_blank_color(self):
        self.node.build(
            clip="CLIP", panel_layout="MASK", strength=1.0, set_cond_area="mask bounds",
            panel_1_prompt="x", panel_1_color="",       # missing color → skip
            panel_2_prompt="y", panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="",  panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        self.assertEqual(len(_CALLS), 1)
        self.assertEqual(_CALLS[0]["mask_color"], "#00FF00")

    def test_no_active_panels_returns_empty_conditioning(self):
        out = self.node.build(
            clip="CLIP", panel_layout="MASK", strength=1.0, set_cond_area="default",
            panel_1_prompt="", panel_1_color="#FF0000",
            panel_2_prompt="", panel_2_color="#00FF00",
            panel_3_prompt="", panel_3_color="#0000FF",
            panel_4_prompt="", panel_4_color="#FFFF00",
            panel_5_prompt="", panel_5_color="#FF00FF",
            panel_6_prompt="", panel_6_color="#00FFFF",
        )
        self.assertEqual(_CALLS, [])
        self.assertEqual(out, ([],))

    def test_strength_and_set_cond_area_propagate(self):
        self.node.build(
            clip="CLIP", panel_layout="MASK", strength=0.6, set_cond_area="default",
            panel_1_prompt="x", panel_1_color="#FF0000",
            panel_2_prompt="",  panel_2_color="#00FF00",
            panel_3_prompt="",  panel_3_color="#0000FF",
            panel_4_prompt="",  panel_4_color="#FFFF00",
            panel_5_prompt="",  panel_5_color="#FF00FF",
            panel_6_prompt="",  panel_6_color="#00FFFF",
        )
        self.assertEqual(_CALLS[0]["strength"], 0.6)
        self.assertEqual(_CALLS[0]["set_cond_area"], "default")


if __name__ == "__main__":
    unittest.main()
