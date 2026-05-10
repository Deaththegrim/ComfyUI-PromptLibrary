"""Sampler SDXL integration tests for the runtime cache registry.

The two 1-slot caches in sampler_sdxl (_UPSCALE_MODEL_CACHE,
_HIRES_CKPT_CACHE) are tuple-or-None rebinds, not dict mutations, so
they need dedicated clear functions rather than `.clear()`. This file
verifies both wire into the runtime unload hook end-to-end.

Skipped when ComfyUI isn't on PYTHONPATH (CI's stdlib-only runner).
sampler_sdxl imports comfy.sample/samplers/sd/utils at module load.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch needed for sampler tests")

try:
    import comfy.sample  # noqa: F401
    _HAS_COMFY = True
except ImportError:
    _HAS_COMFY = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime  # noqa: E402

if _HAS_COMFY:
    import sampler_sdxl  # noqa: E402


def _install_fake_comfy_unload(spy):
    mm_mod = sys.modules.get("comfy.model_management")
    if mm_mod is None:
        comfy_mod = types.ModuleType("comfy")
        mm_mod = types.ModuleType("comfy.model_management")
        comfy_mod.model_management = mm_mod
        sys.modules["comfy"] = comfy_mod
        sys.modules["comfy.model_management"] = mm_mod
    mm_mod.unload_all_models = spy
    return mm_mod


@unittest.skipUnless(_HAS_COMFY, "ComfyUI not on PYTHONPATH")
class SamplerSDXLUnloadHookTests(unittest.TestCase):
    """sampler_sdxl's two 1-slot caches drop when the shared runtime hook
    fires. Mirrors UnloadHookIntegrationTests in test_detailer.py."""

    def setUp(self):
        self._prev_mm = sys.modules.get("comfy.model_management")
        self._prev_upscale = sampler_sdxl._UPSCALE_MODEL_CACHE
        self._prev_hires = sampler_sdxl._HIRES_CKPT_CACHE
        self._prev_installed = runtime._UNLOAD_HOOK_INSTALLED
        runtime._UNLOAD_HOOK_INSTALLED = False

        self._original_called = False

        def _fake_unload(*_a, **_k):
            self._original_called = True

        self._mm_mod = _install_fake_comfy_unload(_fake_unload)

    def tearDown(self):
        sampler_sdxl._UPSCALE_MODEL_CACHE = self._prev_upscale
        sampler_sdxl._HIRES_CKPT_CACHE = self._prev_hires
        runtime._UNLOAD_HOOK_INSTALLED = self._prev_installed
        if self._prev_mm is not None:
            sys.modules["comfy.model_management"] = self._prev_mm

    def test_unload_clears_both_one_slot_caches_and_calls_original(self):
        sampler_sdxl._UPSCALE_MODEL_CACHE = ("4x_upscaler.pth", object())
        sampler_sdxl._HIRES_CKPT_CACHE = ("hires_swap.safetensors", (object(),) * 3)

        runtime.ensure_unload_hook()
        self.assertTrue(runtime._UNLOAD_HOOK_INSTALLED)

        self._mm_mod.unload_all_models()

        self.assertTrue(self._original_called,
                        "wrapper must delegate to the original unload_all_models")
        self.assertIsNone(sampler_sdxl._UPSCALE_MODEL_CACHE)
        self.assertIsNone(sampler_sdxl._HIRES_CKPT_CACHE)


if __name__ == "__main__":
    unittest.main()
