"""Unit tests for GrimmRibbity Character Anchor — wraps IPAdapter Plus.

IPAdapter Plus isn't installed in the test environment, so these tests
mock it via sys.modules injection before character_anchor is imported.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_fake_ipadapter():
    """Inject a fake `custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus` so
    character_anchor's top-level import succeeds. Returns the loader/applier
    classes so tests can spy on them."""
    fake_loader_calls = []
    fake_apply_calls = []

    class FakeUnifiedLoader:
        def load_models(self, model, preset, **kwargs):
            fake_loader_calls.append({"model": model, "preset": preset, "kwargs": kwargs})
            # Returns (model_with_ipadapter, ipadapter_handle)
            return (f"{model}|ipa-loaded({preset})", f"ipadapter-handle({preset})")

    class FakeSimple:
        def apply_ipadapter(self, model, ipadapter, image, weight, start_at, end_at,
                             weight_type, attn_mask=None):
            fake_apply_calls.append({
                "model": model, "ipadapter": ipadapter, "image": image,
                "weight": weight, "start_at": start_at, "end_at": end_at,
                "weight_type": weight_type, "attn_mask": attn_mask,
            })
            return (f"{model}|applied(w={weight},type={weight_type})",)

    pkg_root = types.ModuleType("custom_nodes")
    pkg_ipa = types.ModuleType("custom_nodes.ComfyUI_IPAdapter_plus")
    pkg_module = types.ModuleType("custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus")
    pkg_module.IPAdapterUnifiedLoader = FakeUnifiedLoader
    pkg_module.IPAdapterSimple = FakeSimple
    sys.modules["custom_nodes"] = pkg_root
    sys.modules["custom_nodes.ComfyUI_IPAdapter_plus"] = pkg_ipa
    sys.modules["custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus"] = pkg_module
    return fake_loader_calls, fake_apply_calls


# Install fakes before importing character_anchor
_LOADER_CALLS, _APPLY_CALLS = _install_fake_ipadapter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import character_anchor  # noqa: E402


class CharacterAnchorTests(unittest.TestCase):
    def setUp(self):
        _LOADER_CALLS.clear()
        _APPLY_CALLS.clear()
        self.node = character_anchor.GrimmRibbityCharacterAnchor()

    def test_input_types_shape(self):
        spec = character_anchor.GrimmRibbityCharacterAnchor.INPUT_TYPES()
        self.assertIn("required", spec)
        for k in ("model", "preset", "weight", "weight_type",
                  "start_at", "end_at", "bypass"):
            self.assertIn(k, spec["required"], f"missing required input {k}")
        # reference and attn_mask are optional — reference because bypass=True
        # should work without one wired; attn_mask for the regional bridge.
        self.assertIn("reference", spec["optional"])
        self.assertIn("attn_mask", spec["optional"])
        self.assertEqual(character_anchor.GrimmRibbityCharacterAnchor.RETURN_TYPES, ("MODEL",))

    def test_preset_default_is_plus_face(self):
        spec = character_anchor.GrimmRibbityCharacterAnchor.INPUT_TYPES()
        self.assertEqual(spec["required"]["preset"][1]["default"], "PLUS FACE (portraits)")

    def test_active_path_invokes_loader_and_applier(self):
        out = self.node.anchor(
            model="MODEL_IN", reference="IMG", preset="PLUS FACE (portraits)",
            weight=0.7, weight_type="standard", start_at=0.0, end_at=1.0,
            bypass=False,
        )
        self.assertEqual(len(_LOADER_CALLS), 1)
        self.assertEqual(len(_APPLY_CALLS), 1)
        self.assertEqual(_LOADER_CALLS[0]["preset"], "PLUS FACE (portraits)")
        self.assertEqual(_APPLY_CALLS[0]["weight"], 0.7)
        self.assertEqual(_APPLY_CALLS[0]["weight_type"], "standard")
        # Output shape is the (MODEL,) tuple from IPAdapterSimple
        self.assertEqual(len(out), 1)
        self.assertIn("applied", out[0])

    def test_bypass_returns_model_unchanged_and_skips_ipadapter(self):
        out = self.node.anchor(
            model="MODEL_IN", reference="IMG", preset="PLUS FACE (portraits)",
            weight=0.85, weight_type="style transfer", start_at=0.0, end_at=1.0,
            bypass=True,
        )
        self.assertEqual(out, ("MODEL_IN",))
        self.assertEqual(_LOADER_CALLS, [])
        self.assertEqual(_APPLY_CALLS, [])

    def test_attn_mask_passes_through_when_provided(self):
        self.node.anchor(
            model="MODEL_IN", reference="IMG", preset="STANDARD (medium strength)",
            weight=0.5, weight_type="standard", start_at=0.0, end_at=1.0,
            bypass=False, attn_mask="MASK_OBJ",
        )
        self.assertEqual(_APPLY_CALLS[0]["attn_mask"], "MASK_OBJ")

    def test_attn_mask_default_none_when_omitted(self):
        self.node.anchor(
            model="MODEL_IN", reference="IMG", preset="PLUS (high strength)",
            weight=0.7, weight_type="standard", start_at=0.0, end_at=1.0,
            bypass=False,
        )
        self.assertIsNone(_APPLY_CALLS[0]["attn_mask"])

    def test_bypass_short_circuits_before_loading(self):
        # If IPAdapter Plus had a heavy network/disk side-effect on import,
        # bypass should never trigger it. We assert the loader isn't called.
        for _ in range(5):
            self.node.anchor(
                model=f"M{_}", reference="IMG", preset="VIT-G (medium strength)",
                weight=1.0, weight_type="standard", start_at=0.0, end_at=1.0,
                bypass=True,
            )
        self.assertEqual(_LOADER_CALLS, [])

    def test_weight_window_is_passed_through(self):
        self.node.anchor(
            model="M", reference="IMG", preset="PLUS FACE (portraits)",
            weight=0.6, weight_type="standard", start_at=0.2, end_at=0.85,
            bypass=False,
        )
        call = _APPLY_CALLS[0]
        self.assertEqual(call["start_at"], 0.2)
        self.assertEqual(call["end_at"], 0.85)

    def test_bypass_true_without_reference_passes_through(self):
        # Common case: user wants to disable the IPAdapter for one gen and
        # doesn't want to wire a dummy reference image just to satisfy a
        # required input. With reference now optional, bypass+no-image
        # should work cleanly.
        out = self.node.anchor(
            model="M", preset="PLUS FACE (portraits)",
            weight=0.7, weight_type="standard", start_at=0.0, end_at=1.0,
            bypass=True,  # reference left as default None
        )
        self.assertEqual(out, ("M",))
        self.assertEqual(_LOADER_CALLS, [])

    def test_bypass_false_without_reference_raises(self):
        # The flip side: if you actually want IPAdapter on (bypass=False) but
        # forgot to wire reference, we raise a clear error rather than crash
        # somewhere inside the IPAdapter Plus internals.
        with self.assertRaises(ValueError) as ctx:
            self.node.anchor(
                model="M", preset="PLUS FACE (portraits)",
                weight=0.7, weight_type="standard", start_at=0.0, end_at=1.0,
                bypass=False,  # but no reference!
            )
        self.assertIn("reference", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
