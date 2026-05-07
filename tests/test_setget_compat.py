"""Set/Get node compatibility audit.

kjnodes / rgthree-style Set/Get nodes mirror the connected source's output
type to their own slots — they pass values through untouched and rely on
the type STRING matching between producer and consumer. These tests
verify our nodes' RETURN_TYPES / INPUT_TYPES use stable, matching type
strings so Set→Get round-tripping works without manual type fixes.

Specifically:
  - All custom (non-builtin) type strings are valid identifiers
  - When two nodes share a type (producer.RETURN_TYPES has X and
    consumer.INPUT_TYPES has X), they use the literal-same string
  - No node emits a tuple of mixed/list-typed outputs that Set/Get's
    type mirror would flatten incorrectly
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The producer/consumer round-trip tests import sampler_sdxl + sampler_anima,
# which need `comfy.*` from ComfyUI's path. CI's stdlib-only runner doesn't
# have it. Skip those classes cleanly — the type-string sanity tests in this
# module run regardless.
try:
    import comfy.sample  # noqa: F401
    _HAS_COMFY = True
except ImportError:
    _HAS_COMFY = False
_NEEDS_COMFY = unittest.skipUnless(_HAS_COMFY, "ComfyUI not on PYTHONPATH")

# Standard ComfyUI types — Set/Get always handles these.
_STD_TYPES = {
    "MODEL", "CLIP", "VAE", "IMAGE", "MASK", "CONDITIONING", "LATENT",
    "INT", "STRING", "FLOAT", "BOOLEAN", "COMBO", "AUDIO", "VIDEO",
    "CLIP_VISION", "STYLE_MODEL", "CLIP_VISION_OUTPUT", "CONTROL_NET",
    "GLIGEN", "UPSCALE_MODEL",
}

# Custom types our suite emits. Verified against current code; if you add
# a new one, update this set + add producer/consumer pair to the round-trip
# test below.
_OUR_CUSTOM_TYPES = {
    "SDXL_TUPLE", "GRIMM_SDXL_SCRIPT", "GRIMM_ANIMA_SCRIPT",
}


class TypeStringSanityTests(unittest.TestCase):
    def test_custom_types_are_valid_identifiers(self):
        # Set/Get nodes display the type string as-is in the UI; weird chars
        # render badly. Plain ALL_CAPS_WITH_UNDERSCORES is the convention.
        for t in _OUR_CUSTOM_TYPES:
            self.assertTrue(re.fullmatch(r"[A-Z][A-Z0-9_]+", t),
                             f"custom type {t!r} should be ALL_CAPS_UNDERSCORE")

    def test_no_overlap_with_standard_types(self):
        # If we accidentally collide with a Comfy built-in, Set/Get would
        # round-trip but a future Comfy upgrade could shift semantics.
        self.assertEqual(_OUR_CUSTOM_TYPES & _STD_TYPES, set())


@_NEEDS_COMFY
class ProducerConsumerRoundTripTests(unittest.TestCase):
    """Verify that for each custom type, both ends use the literal string.
    A typo on either side would break Set/Get-mediated wiring even though
    direct wiring would still work (because Comfy's slot-coloring uses the
    string for visual matching but slot connection accepts compatible types).
    """

    def test_sdxl_tuple_pair(self):
        from sampler_sdxl import (GrimmRibbityPackSDXLTuple,
                                    GrimmRibbitySamplerSDXL,
                                    SDXL_TUPLE_TYPE)
        # Producer: Pack SDXL Tuple emits SDXL_TUPLE
        ret_pack = GrimmRibbityPackSDXLTuple.RETURN_TYPES
        self.assertEqual(ret_pack[0], SDXL_TUPLE_TYPE)
        self.assertEqual(SDXL_TUPLE_TYPE, "SDXL_TUPLE")
        # Consumer: SDXL Sampler accepts SDXL_TUPLE
        spec = GrimmRibbitySamplerSDXL.INPUT_TYPES()
        self.assertEqual(spec["required"]["sdxl_tuple"][0], SDXL_TUPLE_TYPE)
        # Producer: SDXL Sampler also re-emits SDXL_TUPLE for chaining
        self.assertIn(SDXL_TUPLE_TYPE, GrimmRibbitySamplerSDXL.RETURN_TYPES)

    def test_grimm_sdxl_script_pair(self):
        from sampler_sdxl import (GrimmRibbityHiResFixScript,
                                    GrimmRibbitySamplerSDXL,
                                    GRIMM_SDXL_SCRIPT_TYPE)
        self.assertEqual(GRIMM_SDXL_SCRIPT_TYPE, "GRIMM_SDXL_SCRIPT")
        self.assertEqual(GrimmRibbityHiResFixScript.RETURN_TYPES[0],
                          GRIMM_SDXL_SCRIPT_TYPE)
        spec = GrimmRibbitySamplerSDXL.INPUT_TYPES()
        # script is in optional inputs
        all_inputs = {**spec.get("required", {}), **spec.get("optional", {}), **spec.get("hidden", {})}
        self.assertEqual(all_inputs["script"][0], GRIMM_SDXL_SCRIPT_TYPE)

    def test_grimm_anima_script_pair(self):
        from sampler_anima import (GrimmRibbityAnimaHiResFixScript,
                                     GrimmRibbityAnimaSampler,
                                     GRIMM_ANIMA_SCRIPT_TYPE)
        self.assertEqual(GRIMM_ANIMA_SCRIPT_TYPE, "GRIMM_ANIMA_SCRIPT")
        self.assertEqual(GrimmRibbityAnimaHiResFixScript.RETURN_TYPES[0],
                          GRIMM_ANIMA_SCRIPT_TYPE)
        spec = GrimmRibbityAnimaSampler.INPUT_TYPES()
        all_inputs = {**spec.get("required", {}), **spec.get("optional", {}), **spec.get("hidden", {})}
        self.assertEqual(all_inputs["script"][0], GRIMM_ANIMA_SCRIPT_TYPE)


class OutputShapeSanityTests(unittest.TestCase):
    """Set/Get can't pass through list-typed outputs cleanly — they
    flatten incorrectly because the SetNode's input slot mirrors the
    string type but doesn't track OUTPUT_IS_LIST. Verify we don't ship
    list outputs anywhere."""

    def _import_node_class(self, modname, classname):
        import importlib
        m = importlib.import_module(modname)
        return getattr(m, classname)

    def test_no_node_emits_list_outputs(self):
        targets = [
            ("character_anchor", "GrimmRibbityCharacterAnchor"),
            ("comic_page",       "GrimmRibbityComicPage"),
            ("style_node",       "PromptLibraryStyle"),
            ("civitai_save",     "GrimmRibbityCivitaiSave"),
            ("lora_picker",      "GrimmRibbityLoraPicker"),
        ]
        for modname, cls in targets:
            try:
                node = self._import_node_class(modname, cls)
            except (ImportError, AttributeError):
                # Some node classes need torch — skip if not importable.
                continue
            list_flag = getattr(node, "OUTPUT_IS_LIST", None)
            if list_flag is not None:
                self.assertFalse(any(list_flag),
                                  f"{cls}.OUTPUT_IS_LIST has True entries → "
                                  f"Set/Get won't pass these cleanly")


@_NEEDS_COMFY
class SetGetSemanticTest(unittest.TestCase):
    """The actual Set/Get behaviour we want: the value flows through
    unchanged. Simulate via direct passthrough — kjnodes' SetNode just
    stores the input and emits it unchanged on its output, then GetNode
    reads from a graph-wide registry by name.

    For custom types like SDXL_TUPLE, the value is a Python tuple; the
    Set/Get path doesn't touch the value, only stores it. Verify the
    tuple shape is preserved across our pack/unpack helpers (which is
    what would actually be stored)."""

    def test_sdxl_tuple_packed_then_unpacked_round_trip(self):
        from sampler_sdxl import _unpack_sdxl_tuple
        # Construct an 8-tuple as Pack SDXL Tuple would emit, then pretend
        # it was Set→Get'd (no transformation), then unpacked by the Sampler.
        original = ("model_obj", "clip_obj", "pos_obj", "neg_obj",
                     None, None, None, None)
        # Set node would store `original` and re-emit it untouched.
        # SetNode-mediated value should round-trip with no shape change.
        unpacked = _unpack_sdxl_tuple(original)
        self.assertEqual(len(unpacked), 8,
                         "unpack must always return 8 elements regardless of "
                         "how many Set/Get hops the tuple took")

    def test_sdxl_tuple_with_full_refiner_round_trip(self):
        from sampler_sdxl import _unpack_sdxl_tuple
        original = ("m", "c", "p", "n", "rm", "rc", "rp", "rn")
        unpacked = _unpack_sdxl_tuple(original)
        self.assertEqual(unpacked, original)


if __name__ == "__main__":
    unittest.main()
