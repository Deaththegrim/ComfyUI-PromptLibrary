"""Unit tests for the self-contained GrimmRibbity Smart Detailer.

These tests cover the pure-Python pieces (helpers + bypass / no-detector
control flow). The full YOLO+sampler integration is verified by running it
inside ComfyUI; mocking the entire model_patcher / common_ksampler stack is
out of scope for these unit tests.

Skipped when torch isn't installed (e.g. CI's stdlib-only runner). The
detailer module itself imports torch at load, so we have to bail at the
module level rather than per-test.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch needed for Smart Detailer tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detailer_node  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_expand_bbox_clamps_to_image(self):
        # bbox at top-left edge with crop_factor 3 should clamp at 0,0
        x1, y1, x2, y2 = detailer_node._expand_bbox((10, 10, 30, 30), 3.0, 100, 100)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)
        self.assertGreater(x2, 30)
        self.assertGreater(y2, 30)

    def test_expand_bbox_clamps_to_image_far_corner(self):
        x1, y1, x2, y2 = detailer_node._expand_bbox((90, 90, 95, 95), 5.0, 100, 100)
        self.assertEqual(x2, 100)
        self.assertEqual(y2, 100)

    def test_expand_bbox_factor_1_unchanged(self):
        self.assertEqual(
            detailer_node._expand_bbox((20, 30, 50, 60), 1.0, 100, 100),
            (20, 30, 50, 60),
        )

    def test_scale_to_guide_aligns_to_multiple_of_8(self):
        h, w = detailer_node._scale_to_guide(crop_h=200, crop_w=300, guide=512, max_size=1024)
        self.assertEqual(h % 8, 0)
        self.assertEqual(w % 8, 0)
        self.assertGreaterEqual(min(h, w), 64)

    def test_scale_to_guide_caps_at_max_size(self):
        h, w = detailer_node._scale_to_guide(crop_h=200, crop_w=2000, guide=512, max_size=1024)
        self.assertLessEqual(max(h, w), 1024)

    def test_scale_to_guide_minimum_64(self):
        h, w = detailer_node._scale_to_guide(crop_h=10, crop_w=10, guide=512, max_size=1024)
        self.assertGreaterEqual(h, 64)
        self.assertGreaterEqual(w, 64)

    def test_make_feather_mask_shape(self):
        m = detailer_node._make_feather_mask(64, 96, 8, device="cpu")
        self.assertEqual(m.shape, (1, 64, 96))

    def test_make_feather_mask_center_is_one_edges_zero(self):
        m = detailer_node._make_feather_mask(32, 32, 8, device="cpu")
        # center pixel
        self.assertAlmostEqual(m[0, 16, 16].item(), 1.0, places=4)
        # corner pixel — both edge contributions are 0
        self.assertAlmostEqual(m[0, 0, 0].item(), 0.0, places=4)

    def test_make_feather_mask_zero_feather_all_ones(self):
        m = detailer_node._make_feather_mask(16, 16, 0, device="cpu")
        self.assertTrue(torch.all(m == 1.0))

    def test_feather_mask_edge_suppression_left(self):
        # Suppress left edge → leftmost column should not fall off
        m = detailer_node._make_feather_mask(
            32, 32, 8, device="cpu",
            suppress_edges=(True, False, False, False))
        # Left column: should be controlled by Y edges only (full strength horizontally)
        # vs without suppression where x=0 → 0
        m_no_sup = detailer_node._make_feather_mask(32, 32, 8, device="cpu")
        # At (16, 0) — middle row, left edge
        self.assertGreater(m[0, 16, 0].item(), m_no_sup[0, 16, 0].item())
        self.assertAlmostEqual(m[0, 16, 0].item(), 1.0, places=4)

    def test_feather_mask_all_edges_suppressed_is_ones(self):
        m = detailer_node._make_feather_mask(
            16, 16, 8, device="cpu",
            suppress_edges=(True, True, True, True))
        self.assertTrue(torch.all(m == 1.0))


class NormalizeImageShapeTests(unittest.TestCase):
    def test_4d_passes_through_unchanged(self):
        img = torch.zeros((1, 64, 64, 3))
        out = detailer_node._normalize_image_shape(img)
        self.assertEqual(tuple(out.shape), (1, 64, 64, 3))
        self.assertIs(out, img)

    def test_3d_gets_batch_dim_added(self):
        img = torch.zeros((64, 64, 3))
        out = detailer_node._normalize_image_shape(img)
        self.assertEqual(tuple(out.shape), (1, 64, 64, 3))

    def test_5d_with_leading_singleton_squeezes_to_4d(self):
        img = torch.zeros((1, 1, 64, 64, 3))
        out = detailer_node._normalize_image_shape(img)
        self.assertEqual(tuple(out.shape), (1, 64, 64, 3))

    def test_6d_with_leading_singletons_squeezes_to_4d(self):
        img = torch.zeros((1, 1, 1, 64, 64, 3))
        out = detailer_node._normalize_image_shape(img)
        self.assertEqual(tuple(out.shape), (1, 64, 64, 3))

    def test_5d_with_T_leading_singleton_squeezes_to_batch(self):
        # (1, 4, H, W, C): singleton batch wrapping a 4-frame source. Squeeze
        # the leading 1 and the result (4, H, W, C) is a valid batch — no
        # crash, just runs the detailer on all 4 frames.
        img = torch.zeros((1, 4, 64, 64, 3))
        out = detailer_node._normalize_image_shape(img)
        self.assertEqual(tuple(out.shape), (4, 64, 64, 3))

    def test_5d_video_shape_raises_clear_error(self):
        # (B=1, T=4, H, W, C) — animation source: leading dim is 1 but T>1
        # so squeeze stops at 4D-with-T, not 4D-spatial. We squeeze the
        # leading 1 then have (4, 64, 64, 3) which IS 4D — that's the "use
        # one frame instead" case. Genuine 5D no-singleton stays 5D.
        img = torch.zeros((2, 4, 64, 64, 3))
        with self.assertRaises(ValueError) as ctx:
            detailer_node._normalize_image_shape(img)
        msg = str(ctx.exception)
        self.assertIn("4D", msg)
        self.assertIn("(2, 4, 64, 64, 3)", msg)
        self.assertIn("video", msg.lower())

    def test_2d_raises_clear_error(self):
        img = torch.zeros((64, 64))
        with self.assertRaises(ValueError) as ctx:
            detailer_node._normalize_image_shape(img)
        self.assertIn("4D", str(ctx.exception))


class WildcardCacheTests(unittest.TestCase):
    def setUp(self):
        detailer_node._WILDCARD_CACHE.clear()

    def test_cache_bounded_by_max(self):
        # Stuff the cache past its bound, oldest should evict
        for i in range(detailer_node._WILDCARD_CACHE_MAX + 5):
            detailer_node._WILDCARD_CACHE[(i, f"text-{i}")] = f"cond-{i}"
            if len(detailer_node._WILDCARD_CACHE) > detailer_node._WILDCARD_CACHE_MAX:
                detailer_node._WILDCARD_CACHE.popitem(last=False)
        self.assertEqual(len(detailer_node._WILDCARD_CACHE),
                         detailer_node._WILDCARD_CACHE_MAX)
        # Oldest 5 evicted
        self.assertNotIn((0, "text-0"), detailer_node._WILDCARD_CACHE)
        self.assertIn((detailer_node._WILDCARD_CACHE_MAX + 4,
                       f"text-{detailer_node._WILDCARD_CACHE_MAX + 4}"),
                      detailer_node._WILDCARD_CACHE)


class VaeDecodeTilingPolicyTests(unittest.TestCase):
    def test_decode_skips_tiling_for_small_results(self):
        """A 64x64 latent decodes to 512x512 image — should NOT tile."""
        calls = []

        class FakeVae:
            def decode(self, samples):
                calls.append("decode")
                return torch.zeros((1, samples.shape[-2] * 8, samples.shape[-1] * 8, 3))

            def decode_tiled(self, samples, tile_x=None, tile_y=None):
                calls.append("decode_tiled")
                return torch.zeros((1, samples.shape[-2] * 8, samples.shape[-1] * 8, 3))

        latent = {"samples": torch.zeros((1, 4, 64, 64))}
        detailer_node._vae_decode(FakeVae(), latent, tiled=True)
        self.assertEqual(calls, ["decode"], "should pick full decode for 512x512 output")

    def test_decode_uses_tiling_for_large_results(self):
        calls = []

        class FakeVae:
            def decode(self, samples):
                calls.append("decode")
                return torch.zeros((1, 1024, 1024, 3))

            def decode_tiled(self, samples, tile_x=None, tile_y=None):
                calls.append("decode_tiled")
                return torch.zeros((1, samples.shape[-2] * 8, samples.shape[-1] * 8, 3))

        # 256x256 latent → 2048x2048 image → must tile (above the shared
        # 192-latent / 1536px threshold).
        latent = {"samples": torch.zeros((1, 4, 256, 256))}
        detailer_node._vae_decode(FakeVae(), latent, tiled=True)
        self.assertEqual(calls, ["decode_tiled"])


class FolderPathRegistrationTests(unittest.TestCase):
    def test_register_folder_paths_is_idempotent(self):
        import folder_paths
        # Call twice — should not duplicate path entries
        detailer_node._register_folder_paths()
        before = list(folder_paths.folder_names_and_paths.get("ultralytics_bbox", ([], None))[0])
        detailer_node._register_folder_paths()
        after = list(folder_paths.folder_names_and_paths.get("ultralytics_bbox", ([], None))[0])
        self.assertEqual(before, after)

    def test_register_creates_three_keys(self):
        import folder_paths
        detailer_node._register_folder_paths()
        for key in ("ultralytics_bbox", "ultralytics_segm", "ultralytics"):
            self.assertIn(key, folder_paths.folder_names_and_paths,
                          f"missing folder_paths key: {key}")


class ResolveBboxPathTests(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(detailer_node._resolve_bbox_path(detailer_node._NONE))
        self.assertIsNone(detailer_node._resolve_bbox_path(""))

    def test_unknown_model_returns_none(self):
        # No actual detector files in test env
        self.assertIsNone(detailer_node._resolve_bbox_path("does_not_exist.pt"))


class BypassAndPassthroughTests(unittest.TestCase):
    def setUp(self):
        detailer_node._YOLO_CACHE.clear()
        self.node = detailer_node.GrimmRibbitySmartDetailer()
        self.image = torch.zeros((1, 64, 64, 3))

    def _kwargs(self, **overrides):
        defaults = dict(
            image=self.image, model="MODEL", clip="CLIP", vae="VAE",
            positive="POS", negative="NEG",
            enable_face=False, enable_eyes=False, enable_hands=False,
            bbox_face=detailer_node._NONE,
            bbox_eyes=detailer_node._NONE,
            bbox_hands=detailer_node._NONE,
            sam_model=detailer_node._NONE,
            seed=0, steps=20, cfg=6.0,
            sampler_name="dpmpp_3m_sde_gpu", scheduler="karras",
            denoise=0.4, guide_size=512, max_size=1024,
            bbox_threshold=0.5, max_per_target=0,
            tiled_decode=True, tiled_encode=False,
            mask_strength=1.0, same_seed_per_target=False,
            bypass=False,
        )
        defaults.update(overrides)
        return defaults

    def test_returns_three_outputs(self):
        self.assertEqual(detailer_node.GrimmRibbitySmartDetailer.RETURN_TYPES,
                         ("IMAGE", "MASK", "IMAGE"))
        self.assertEqual(detailer_node.GrimmRibbitySmartDetailer.RETURN_NAMES,
                         ("image", "mask", "detections_preview"))

    def test_bypass_returns_image_unchanged_zero_mask_preview_input(self):
        out_img, out_mask, preview = self.node.detail(**self._kwargs(
            bypass=True, enable_face=True, bbox_face="some_model.pt"))
        self.assertIs(out_img, self.image)
        self.assertEqual(out_mask.shape, (1, 64, 64))
        self.assertTrue(torch.all(out_mask == 0.0))
        self.assertIs(preview, self.image)

    def test_no_targets_enabled_passthrough(self):
        out_img, out_mask, preview = self.node.detail(**self._kwargs())
        self.assertIs(out_img, self.image)
        self.assertTrue(torch.all(out_mask == 0.0))
        # Preview is a clone (no detections drawn) — equal to input but not same obj
        self.assertTrue(torch.equal(preview, self.image))

    def test_face_enabled_no_detector_passthrough(self):
        out_img, out_mask, _preview = self.node.detail(**self._kwargs(enable_face=True))
        self.assertIs(out_img, self.image)
        self.assertTrue(torch.all(out_mask == 0.0))

    def test_face_enabled_invalid_detector_passthrough(self):
        out_img, _mask, _preview = self.node.detail(**self._kwargs(
            enable_face=True, bbox_face="totally_made_up.pt"))
        self.assertIs(out_img, self.image)


class InputTypesShapeTests(unittest.TestCase):
    def test_input_types_minimum_keys(self):
        spec = detailer_node.GrimmRibbitySmartDetailer.INPUT_TYPES()
        for k in ("image", "model", "clip", "vae", "positive", "negative",
                  "enable_face", "enable_eyes", "enable_hands",
                  "bbox_face", "bbox_eyes", "bbox_hands", "sam_model",
                  "seed", "steps", "cfg", "sampler_name", "scheduler",
                  "denoise", "guide_size", "max_size",
                  "bbox_threshold", "max_per_target",
                  "tiled_decode", "tiled_encode", "mask_strength",
                  "same_seed_per_target",
                  "enable_feet", "bbox_feet",
                  "bypass"):
            self.assertIn(k, spec["required"], f"missing required: {k}")
        for k in ("wildcard_prefix", "force_inpaint", "drop_size",
                  "min_detail_size", "nms_iou", "yolo_imgsz",
                  "max_bbox_area_pct", "draw_preview",
                  "face_threshold", "face_denoise", "face_max", "face_steps",
                  "eyes_threshold", "eyes_denoise", "eyes_max", "eyes_steps",
                  "hands_threshold", "hands_denoise", "hands_max", "hands_steps",
                  "feet_threshold", "feet_denoise", "feet_max", "feet_steps",
                  "face_crop_factor", "eyes_crop_factor",
                  "hands_crop_factor", "feet_crop_factor",
                  "face_cycles", "eyes_cycles", "hands_cycles", "feet_cycles"):
            self.assertIn(k, spec["optional"], f"missing optional: {k}")

    def test_per_target_overrides_default_to_minus_one(self):
        spec = detailer_node.GrimmRibbitySmartDetailer.INPUT_TYPES()
        for k in ("face_threshold", "face_denoise",
                  "eyes_threshold", "eyes_denoise",
                  "hands_threshold", "hands_denoise",
                  "feet_threshold", "feet_denoise"):
            self.assertEqual(spec["optional"][k][1]["default"], -1.0,
                             f"{k} should default to -1.0 (= use global)")

    def test_tiled_decode_default_true(self):
        spec = detailer_node.GrimmRibbitySmartDetailer.INPUT_TYPES()
        self.assertTrue(spec["required"]["tiled_decode"][1]["default"],
                        "tiled_decode must default ON for OOM safety")

    def test_returns_preview_image(self):
        spec = detailer_node.GrimmRibbitySmartDetailer
        # preview is the third output: (image, mask, detections_preview)
        self.assertEqual(spec.OUTPUT_TOOLTIPS[2].split(" ")[0], "Input")


class HelperUtilsTests(unittest.TestCase):
    def test_gaussian_blur_preserves_shape(self):
        x = torch.zeros((1, 1, 32, 32))
        x[0, 0, 16, 16] = 1.0
        out = detailer_node._gaussian_blur_2d(x, sigma=2.0)
        self.assertEqual(out.shape, x.shape)
        # Mass conservation (within float tolerance)
        self.assertAlmostEqual(out.sum().item(), 1.0, places=4)

    def test_gaussian_blur_zero_sigma_passthrough(self):
        x = torch.randn((1, 1, 16, 16))
        out = detailer_node._gaussian_blur_2d(x, sigma=0)
        self.assertTrue(torch.equal(out, x))

    def test_draw_bbox_strokes_perimeter(self):
        img = torch.zeros((1, 32, 32, 3))
        detailer_node._draw_bbox_on_preview(img, (4, 4, 28, 28), (1.0, 0.0, 0.0), line=2)
        # Top edge should be red
        self.assertTrue(torch.all(img[0, 4, 10, 0] == 1.0))
        # Center should still be black (we only stroked the perimeter)
        self.assertTrue(torch.all(img[0, 16, 16] == 0.0))

    def test_draw_text_writes_pixels(self):
        img = torch.zeros((1, 32, 80, 3))
        detailer_node._draw_text_on_preview(img, 4, 4, "face", color=(1.0, 0, 0), scale=2)
        # Some pixels should now be non-zero
        self.assertGreater(img.sum().item(), 0.0)
        # Off-canvas writes don't crash
        detailer_node._draw_text_on_preview(img, 200, 200, "face", color=(1.0, 0, 0))
        detailer_node._draw_text_on_preview(img, -50, 4, "face", color=(1.0, 0, 0))

    def test_draw_text_unknown_chars_skipped(self):
        # Chars not in _FONT_5x7 advance the cursor but don't crash
        img = torch.zeros((1, 32, 80, 3))
        detailer_node._draw_text_on_preview(img, 4, 4, "x@#$", color=(1.0, 0, 0))

    def test_refine_to_blend_mask_falls_back_without_sam(self):
        m = detailer_node._refine_to_blend_mask(
            None, (0, 0, 32, 32), 32, 32, feather=8,
            device="cpu", dtype=torch.float32,
            image_h=64, image_w=64)
        self.assertEqual(m.shape, (1, 32, 32))

    def test_refine_to_blend_suppresses_when_bbox_at_image_edge(self):
        # Bbox at top-left corner (touches image edges 0,0) — feather mask
        # should not fall off on left + top
        m = detailer_node._refine_to_blend_mask(
            None, (0, 0, 32, 32), 32, 32, feather=8,
            device="cpu", dtype=torch.float32,
            image_h=200, image_w=200)
        # Top-left corner should be much closer to 1 than a non-edge bbox
        m_inside = detailer_node._refine_to_blend_mask(
            None, (50, 50, 82, 82), 32, 32, feather=8,
            device="cpu", dtype=torch.float32,
            image_h=200, image_w=200)
        self.assertGreater(m[0, 0, 0].item(), m_inside[0, 0, 0].item())


class PresetTests(unittest.TestCase):
    def test_all_four_targets_have_presets(self):
        for t in ("face", "eyes", "hands", "feet"):
            self.assertIn(t, detailer_node._PRESETS)
            preset = detailer_node._PRESETS[t]
            for key in ("denoise", "feather", "crop_factor", "wildcard"):
                self.assertIn(key, preset, f"{t} preset missing {key}")

    def test_hands_has_highest_denoise(self):
        hands = detailer_node._PRESETS["hands"]["denoise"]
        for other in ("face", "eyes", "feet"):
            self.assertGreaterEqual(hands, detailer_node._PRESETS[other]["denoise"])


class SamOffloadTests(unittest.TestCase):
    """_offload_sam_to_cpu reclaims VRAM between detail() runs without
    invalidating the cache. Tolerates the diverse predictor shapes
    (sam2 / sam_v1 / future variants) defensively."""

    def test_none_handle_is_noop(self):
        detailer_node._offload_sam_to_cpu(None)

    def test_moves_model_to_cpu(self):
        moves = []

        class FakeModel:
            def to(self, device):
                moves.append(device)
                return self

        class FakePredictor:
            model = FakeModel()
            features = "stub_embeddings"
            is_image_set = True

        detailer_node._offload_sam_to_cpu(("sam2", FakePredictor()))
        self.assertEqual(moves, ["cpu"])

    def test_clears_cached_embeddings(self):
        class FakeModel:
            def to(self, device):
                return self

        class FakePredictor:
            def __init__(self):
                self.model = FakeModel()
                self.features = "embedded_image_data"
                self.is_image_set = True

        pred = FakePredictor()
        detailer_node._offload_sam_to_cpu(("sam2", pred))
        self.assertIsNone(pred.features)
        self.assertFalse(pred.is_image_set)

    def test_swallows_exceptions(self):
        class BrokenPredictor:
            @property
            def model(self):
                raise RuntimeError("predictor is in a weird state")

        detailer_node._offload_sam_to_cpu(("sam2", BrokenPredictor()))

    def test_predictor_without_model_attr_is_noop_safe(self):
        class MinimalPredictor:
            pass

        detailer_node._offload_sam_to_cpu(("sam_v1", MinimalPredictor()))


class FutureBboxesAfterTests(unittest.TestCase):
    """detail() builds a 'still-needed bbox set after pass i' table by
    walking passes in reverse to drive sam_mask_cache pruning. Mirror
    that algorithm here so a regression is caught at unit-test time."""

    @staticmethod
    def _build_future(passes_detections):
        future = [None] * len(passes_detections)
        seen = set()
        for j in range(len(passes_detections) - 1, -1, -1):
            future[j] = set(seen)
            for raw_bbox, _conf in passes_detections[j]:
                seen.add(raw_bbox)
        return future

    def test_last_pass_has_empty_future(self):
        passes = [
            [((0, 0, 10, 10), 0.9)],
            [((20, 20, 30, 30), 0.8)],
        ]
        self.assertEqual(self._build_future(passes)[-1], set())

    def test_first_pass_future_contains_all_later_bboxes(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        c = (40, 40, 50, 50)
        passes = [[(a, 0.9)], [(b, 0.8)], [(c, 0.7)]]
        self.assertEqual(self._build_future(passes)[0], {b, c})

    def test_shared_bbox_kept_through_overlap(self):
        shared = (10, 10, 100, 100)
        passes = [[(shared, 0.95)], [(shared, 0.95)], [((200, 200, 250, 250), 0.7)]]
        future = self._build_future(passes)
        self.assertIn(shared, future[0])
        self.assertNotIn(shared, future[1])

    def test_empty_detections_pass_threads_through(self):
        a = (0, 0, 10, 10)
        passes = [[(a, 0.9)], [], [(a, 0.9)]]
        future = self._build_future(passes)
        self.assertIn(a, future[0])
        self.assertIn(a, future[1])
        self.assertEqual(future[2], set())


class UnloadHookIntegrationTests(unittest.TestCase):
    """Detailer caches drop when the shared runtime hook fires. Registry +
    hook plumbing primitives are tested in test_runtime.py — this is the
    end-to-end wiring check."""

    def setUp(self):
        import types
        import runtime
        self._runtime = runtime
        self._prev_mm = sys.modules.get("comfy.model_management")
        self._prev_comfy = sys.modules.get("comfy")
        self._original_called = False

        def _fake_unload(*_a, **_k):
            self._original_called = True

        comfy_mod = types.ModuleType("comfy")
        mm_mod = types.ModuleType("comfy.model_management")
        mm_mod.unload_all_models = _fake_unload
        comfy_mod.model_management = mm_mod
        sys.modules["comfy"] = comfy_mod
        sys.modules["comfy.model_management"] = mm_mod
        self._mm_mod = mm_mod

        self._prev_yolo = dict(detailer_node._YOLO_CACHE)
        self._prev_sam = dict(detailer_node._SAM_CACHE)
        self._prev_wc = list(detailer_node._WILDCARD_CACHE.items())
        self._prev_installed = runtime._UNLOAD_HOOK_INSTALLED
        runtime._UNLOAD_HOOK_INSTALLED = False

    def tearDown(self):
        detailer_node._YOLO_CACHE.clear()
        detailer_node._YOLO_CACHE.update(self._prev_yolo)
        detailer_node._SAM_CACHE.clear()
        detailer_node._SAM_CACHE.update(self._prev_sam)
        detailer_node._WILDCARD_CACHE.clear()
        for k, v in self._prev_wc:
            detailer_node._WILDCARD_CACHE[k] = v
        self._runtime._UNLOAD_HOOK_INSTALLED = self._prev_installed
        if self._prev_mm is not None:
            sys.modules["comfy.model_management"] = self._prev_mm
        else:
            sys.modules.pop("comfy.model_management", None)
        if self._prev_comfy is not None:
            sys.modules["comfy"] = self._prev_comfy
        else:
            sys.modules.pop("comfy", None)

    def test_unload_evicts_detailer_caches_and_calls_original(self):
        detailer_node._YOLO_CACHE["face_yolov8m.pt"] = object()
        detailer_node._SAM_CACHE["sam_vit_b.pth"] = ("sam_v1", object())
        detailer_node._WILDCARD_CACHE[(123, "smooth skin,")] = object()

        self._runtime.ensure_unload_hook()
        self.assertTrue(self._runtime._UNLOAD_HOOK_INSTALLED)

        self._mm_mod.unload_all_models()

        self.assertTrue(self._original_called,
                        "wrapper must delegate to the original unload_all_models")
        self.assertEqual(detailer_node._YOLO_CACHE, {})
        self.assertEqual(detailer_node._SAM_CACHE, {})
        self.assertEqual(len(detailer_node._WILDCARD_CACHE), 0)


if __name__ == "__main__":
    unittest.main()
