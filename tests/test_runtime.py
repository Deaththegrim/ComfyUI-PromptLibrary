"""Unit tests for the shared runtime helpers (registry + lazy unload hook).

End-to-end wiring per consumer module is tested in their own test files
(e.g. test_detailer.py::UnloadHookIntegrationTests). This file only
covers the primitives in isolation."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="runtime imports torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: E402


def _install_fake_comfy(unload_fn):
    comfy_mod = types.ModuleType("comfy")
    mm_mod = types.ModuleType("comfy.model_management")
    mm_mod.unload_all_models = unload_fn
    comfy_mod.model_management = mm_mod
    sys.modules["comfy"] = comfy_mod
    sys.modules["comfy.model_management"] = mm_mod
    return mm_mod


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._prev_clearers = list(runtime._CACHE_CLEARERS)
        self._prev_installed = runtime._UNLOAD_HOOK_INSTALLED
        self._prev_mm = sys.modules.get("comfy.model_management")
        self._prev_comfy = sys.modules.get("comfy")
        runtime._CACHE_CLEARERS.clear()
        runtime._UNLOAD_HOOK_INSTALLED = False

    def tearDown(self):
        runtime._CACHE_CLEARERS[:] = self._prev_clearers
        runtime._UNLOAD_HOOK_INSTALLED = self._prev_installed
        if self._prev_mm is not None:
            sys.modules["comfy.model_management"] = self._prev_mm
        else:
            sys.modules.pop("comfy.model_management", None)
        if self._prev_comfy is not None:
            sys.modules["comfy"] = self._prev_comfy
        else:
            sys.modules.pop("comfy", None)

    def test_register_cache_appends_clearer(self):
        runtime.register_cache(lambda: None)
        runtime.register_cache(lambda: None)
        self.assertEqual(len(runtime._CACHE_CLEARERS), 2)

    def test_evict_runs_every_registered_clearer(self):
        calls = []
        runtime.register_cache(lambda: calls.append("a"))
        runtime.register_cache(lambda: calls.append("b"))
        runtime._evict_all()
        self.assertEqual(calls, ["a", "b"])

    def test_evict_keeps_going_when_one_clearer_raises(self):
        calls = []

        def boom():
            raise RuntimeError("intentional")

        runtime.register_cache(lambda: calls.append("first"))
        runtime.register_cache(boom)
        runtime.register_cache(lambda: calls.append("third"))
        runtime._evict_all()
        self.assertEqual(calls, ["first", "third"])

    def test_install_is_idempotent(self):
        called = []
        mm = _install_fake_comfy(lambda: called.append("orig"))
        runtime.ensure_unload_hook()
        wrapped_once = mm.unload_all_models
        runtime.ensure_unload_hook()
        self.assertIs(mm.unload_all_models, wrapped_once,
                      "second install must not re-wrap the already-wrapped fn")

    def test_install_silent_when_comfy_missing(self):
        # Setting sys.modules entries to None forces ImportError even when
        # comfy is importable from disk (locally with PYTHONPATH set to a
        # ComfyUI checkout). Plain pop() would let the import succeed and
        # the test would assert wrong.
        sys.modules["comfy"] = None
        sys.modules["comfy.model_management"] = None
        runtime.ensure_unload_hook()
        self.assertFalse(runtime._UNLOAD_HOOK_INSTALLED)

    def test_install_then_unload_runs_clearers_then_original(self):
        order = []
        runtime.register_cache(lambda: order.append("cleared"))
        mm = _install_fake_comfy(lambda: order.append("original"))
        runtime.ensure_unload_hook()
        mm.unload_all_models()
        self.assertEqual(order, ["cleared", "original"])

    def test_hip_sync_no_op_path_does_not_raise(self):
        # Whether running on CUDA proper, HIP, or CPU-only torch, hip_sync
        # must not raise. The HIP path's torch.cuda.synchronize is itself
        # wrapped in try/except; the no-op CPU path returns early.
        runtime.hip_sync("test")


if __name__ == "__main__":
    unittest.main()
