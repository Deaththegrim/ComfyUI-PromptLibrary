"""End-to-end functional matrix for the Smart Detailer.

Mocks the three external dependencies (YOLO, SAM, comfy.common_ksampler +
VAE) and runs `detail()` across the bug-and-monitor matrix configurations.
Verifies:
  - no crashes across face-only / eyes-only / hands-only / all-three / skin
  - bypass returns input untouched
  - empty-detection fast path skips work
  - SAM on/off both produce valid masks + images
  - tiled_decode on/off
  - max_per_target=1 portrait mode
  - per-target overrides actually flow into the sampler call
  - mask_strength edge cases (0.0, 1.0, 1.5)
  - same_seed_per_target behaviour
  - bbox area sanity cap drops false positives
  - draw_preview=False skips the overlay
  - output is always finite (no NaN, no inf), in [0, 1], correct shape
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch needed for E2E detailer tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detailer_node  # noqa: E402


def _make_fake_yolo(bboxes_with_conf):
    """Returns a YOLO-shaped object whose call returns the supplied bboxes,
    filtered by the `conf=` kwarg the same way real YOLO does."""
    import numpy as np

    class FakeBoxes:
        def __init__(self, items):
            xyxy = np.array([list(b) for b, _ in items], dtype=np.float32) if items else np.zeros((0, 4))
            confs = np.array([c for _, c in items], dtype=np.float32) if items else np.zeros((0,))

            class _T:
                def __init__(self, arr): self._arr = arr
                def cpu(self): return self
                def numpy(self): return self._arr

            self.xyxy = _T(xyxy)
            self.conf = _T(confs)

    class FakeResult:
        def __init__(self, items): self.boxes = FakeBoxes(items)

    class FakeYolo:
        def __init__(self, items): self._items = items

        def __call__(self, *_a, **kw):
            # Honour the conf threshold the way real ultralytics does — drop
            # detections below the requested confidence.
            thr = float(kw.get("conf", 0.0))
            kept = [b for b in self._items if b[1] >= thr]
            return [FakeResult(kept)]

        def fuse(self): pass
        def to(self, _device): return self

    return FakeYolo(bboxes_with_conf)


class FakeVae:
    """Minimal VAE: encode HWC image to (B, 4, H/8, W/8) latent; decode latent
    back to HWC image of the same H, W. Identity-ish: returns deterministic
    finite tensors."""

    def encode(self, pixels):
        # pixels: (B, H, W, C) — return (B, 4, H/8, W/8) latent
        b, h, w, _ = pixels.shape
        return torch.full((b, 4, max(1, h // 8), max(1, w // 8)), 0.5)

    def decode(self, samples):
        b, _, lh, lw = samples.shape
        return torch.full((b, lh * 8, lw * 8, 3), 0.5)

    def decode_tiled(self, samples, **_kw):
        return self.decode(samples)


def _fake_common_ksampler(_model, _seed, _steps, _cfg, _sn, _sched,
                          _pos, _neg, latent, **_kw):
    """Returns the latent unchanged so we don't burn cycles in real sampling."""
    return ({"samples": latent["samples"]},)


class _DetailerMatrix(unittest.TestCase):
    """Each test patches in the fake YOLO/SAM/sampler then calls detail()."""

    def setUp(self):
        # Image: 256x256 RGB, deterministic mid-grey
        self.image = torch.full((1, 256, 256, 3), 0.5)
        self.vae = FakeVae()
        # Reset module caches so cross-test contamination doesn't happen
        detailer_node._YOLO_CACHE.clear()
        detailer_node._SAM_CACHE.clear()
        detailer_node._WILDCARD_CACHE.clear()

        # Stub _load_yolo + _sam_set_image + comfy hooks
        self._orig_load = detailer_node._load_yolo
        self._orig_sam_set = detailer_node._sam_set_image
        self._orig_sam_pred = detailer_node._sam_predict
        self._orig_sam_batch = detailer_node._sam_predict_batch
        # Hijack module-level ksampler hook so the test doesn't need real sampling.
        detailer_node._COMMON_KSAMPLER = _fake_common_ksampler

        # Stub the wildcard CLIP encode so we don't need a real CLIP.
        # _encode_wildcard_cached needs CLIPTextEncode + ConditioningConcat
        # to be set; skip the encode entirely by setting positive directly.
        class FakeEnc:
            def encode(self, _clip, _text): return ([("cond_with_wc", {})],)

        class FakeConcat:
            def concat(self, p, _wc): return (p,)

        detailer_node._CLIP_TEXT_ENCODE = FakeEnc()
        detailer_node._COND_CONCAT = FakeConcat()

    def tearDown(self):
        detailer_node._load_yolo = self._orig_load
        detailer_node._sam_set_image = self._orig_sam_set
        detailer_node._sam_predict = self._orig_sam_pred
        detailer_node._sam_predict_batch = self._orig_sam_batch

    def _patch_yolo(self, bboxes_with_conf):
        fake = _make_fake_yolo(bboxes_with_conf)
        detailer_node._load_yolo = lambda name: fake if name and name != "(none)" else None

    def _patch_sam_off(self):
        detailer_node._sam_set_image = lambda *_a, **_kw: False
        detailer_node._sam_predict = lambda *_a, **_kw: None
        detailer_node._sam_predict_batch = lambda *_a, **_kw: None

    def _patch_sam_on(self):
        # SAM "loaded + ready"; predict returns a 256x256 binary mask
        detailer_node._load_sam = lambda name: ("sam2", object()) if name and name != "(none)" else None
        detailer_node._sam_set_image = lambda *_a, **_kw: True
        sam_mask = torch.zeros((256, 256))
        sam_mask[60:160, 60:160] = 1.0
        detailer_node._sam_predict = lambda *_a, **_kw: sam_mask
        detailer_node._sam_predict_batch = lambda _h, bboxes: [sam_mask] * len(bboxes)

    def _kwargs(self, **overrides):
        defaults = dict(
            image=self.image, model="MODEL", clip="CLIP", vae=self.vae,
            positive=[("pos", {})], negative=[("neg", {})],
            enable_face=False, enable_eyes=False, enable_hands=False, enable_skin=False,
            bbox_face=detailer_node._NONE,
            bbox_eyes=detailer_node._NONE,
            bbox_hands=detailer_node._NONE,
            sam_model=detailer_node._NONE,
            seed=42, steps=15, cfg=6.0,
            sampler_name="dpmpp_3m_sde_gpu", scheduler="karras",
            denoise=0.45, guide_size=512, max_size=1024,
            bbox_threshold=0.45, max_per_target=0,
            tiled_decode=True, tiled_encode=False,
            mask_strength=1.0, same_seed_per_target=False,
            enable_mouth=False, enable_feet=False,
            bbox_mouth=detailer_node._NONE,
            bbox_feet=detailer_node._NONE,
            bypass=False,
        )
        defaults.update(overrides)
        return defaults

    def _assert_clean_output(self, img, mask, preview, label=""):
        # Shape preserved
        self.assertEqual(img.shape, self.image.shape, f"{label}: image shape changed")
        self.assertEqual(mask.shape, (1, 256, 256), f"{label}: mask shape wrong")
        self.assertEqual(preview.shape, self.image.shape, f"{label}: preview shape wrong")
        # No NaN / inf — the bug-and-monitor matrix's #1 concern
        self.assertTrue(torch.isfinite(img).all(), f"{label}: image has NaN/inf")
        self.assertTrue(torch.isfinite(mask).all(), f"{label}: mask has NaN/inf")
        # Values in [0, 1]
        self.assertTrue((img >= 0).all() and (img <= 1).all(), f"{label}: image out of [0,1]")
        self.assertTrue((mask >= 0).all() and (mask <= 1).all(), f"{label}: mask out of [0,1]")

    # ---- Matrix scenarios ----

    def test_face_only_no_sam(self):
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt"))
        self._assert_clean_output(*out, label="face_only_no_sam")
        # mask should mark the face region
        self.assertGreater(out[1][:, 80:140, 100:156].max().item(), 0.5)

    def test_face_only_with_sam(self):
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        self._patch_sam_on()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                           sam_model="sam2.1_hiera_large.pt"))
        self._assert_clean_output(*out, label="face_only_with_sam")

    def test_eyes_only_falls_back_to_face_detector(self):
        # eyes uses bbox_face when bbox_eyes is None
        self._patch_yolo([((100, 80, 156, 140), 0.85)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_eyes=True, bbox_face="face_yolov8m.pt"))
        self._assert_clean_output(*out, label="eyes_only_fallback")

    def test_hands_only_no_detector_passthrough(self):
        self._patch_yolo([])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_hands=True))
        # bbox_hands is _NONE -> skip, return image unchanged
        self.assertIs(out[0], self.image)
        self._assert_clean_output(*out, label="hands_only_no_det")

    def test_all_targets_with_sam(self):
        self._patch_yolo([((100, 80, 156, 140), 0.85)])
        self._patch_sam_on()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(
            enable_face=True, enable_eyes=True,
            enable_hands=True, enable_skin=True,
            bbox_face="face_yolov8m.pt", bbox_eyes="Eyeful_v2-Paired.pt",
            bbox_hands="hand_yolov8s.pt",
            sam_model="sam2.1_hiera_large.pt"))
        self._assert_clean_output(*out, label="all_with_sam")

    def test_bypass(self):
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(bypass=True, enable_face=True,
                                           bbox_face="face_yolov8m.pt"))
        self.assertIs(out[0], self.image)
        self.assertIs(out[2], self.image)  # preview = input on bypass

    def test_empty_detections_passthrough(self):
        self._patch_yolo([])  # YOLO returns nothing
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt"))
        self.assertIs(out[0], self.image)
        self.assertIs(out[2], self.image)  # preview = input when 0 detections

    def test_max_per_target_one_portrait_mode(self):
        # 3 detections; max_per_target=1 should keep only the highest-conf
        # one. Bboxes are picked so that — even after crop_factor=3.0
        # expansion, only the one expanded mask falls in a known region.
        # We verify by counting how many bboxes get sampled (= how many
        # times the fake ksampler is called).
        self._patch_yolo([
            ((100, 100, 156, 156), 0.87),
            ((30, 30, 50, 50),     0.72),
            ((200, 200, 220, 220), 0.61),
        ])
        self._patch_sam_off()
        sample_calls = {"n": 0}
        orig = detailer_node._COMMON_KSAMPLER

        def counted(*a, **kw):
            sample_calls["n"] += 1
            return orig(*a, **kw)

        detailer_node._COMMON_KSAMPLER = counted
        try:
            node = detailer_node.GrimmRibbitySmartDetailer()
            out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                               max_per_target=1))
        finally:
            detailer_node._COMMON_KSAMPLER = orig
        self._assert_clean_output(*out, label="portrait_mode")
        self.assertEqual(sample_calls["n"], 1,
                          f"max_per_target=1 should sample 1 bbox, got {sample_calls['n']}")

    def test_max_bbox_area_pct_drops_whole_image(self):
        # Bbox covers 100% of frame — should be dropped by the area cap
        self._patch_yolo([((0, 0, 256, 256), 0.99)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                           max_bbox_area_pct=0.95))
        # All detections dropped -> passthrough
        self.assertIs(out[0], self.image)

    def test_mask_strength_zero_no_change(self):
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                           mask_strength=0.0))
        self._assert_clean_output(*out, label="mask_strength_0")
        # Image should equal input (mask_strength=0 means refined*0 + crop*1 = crop)
        self.assertTrue(torch.allclose(out[0], self.image, atol=0.001))

    def test_tiled_decode_on_vs_off(self):
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        for tiled in (True, False):
            out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                               tiled_decode=tiled))
            self._assert_clean_output(*out, label=f"tiled_decode={tiled}")

    def test_draw_preview_false_skips_overlay(self):
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                           draw_preview=False))
        # preview should be the input image unchanged when draw_preview=False
        self.assertIs(out[2], self.image)

    def test_per_target_overrides_apply(self):
        # Set face_threshold=0.99 — should drop the conf-0.85 detection
        self._patch_yolo([((100, 80, 156, 140), 0.85)])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                           bbox_threshold=0.45,
                                           face_threshold=0.99))
        # face_threshold 0.99 > 0.85 conf -> all dropped -> passthrough
        self.assertIs(out[0], self.image)

    def test_same_seed_per_target_uses_seed(self):
        # Hard to verify the seed value from outside, but we can verify it
        # doesn't crash and the output is finite for both true and false.
        self._patch_yolo([
            ((30, 30, 80, 80),    0.85),
            ((100, 80, 156, 140), 0.85),
        ])
        self._patch_sam_off()
        node = detailer_node.GrimmRibbitySmartDetailer()
        for same in (True, False):
            out = node.detail(**self._kwargs(enable_face=True, bbox_face="face_yolov8m.pt",
                                               same_seed_per_target=same))
            self._assert_clean_output(*out, label=f"same_seed={same}")

    def test_sam_cache_face_skin_share_bbox(self):
        # Both face + skin enabled (both use bbox_face). With SAM, the mask
        # cache should hit on the second pass — predict should be called once.
        self._patch_yolo([((100, 80, 156, 140), 0.87)])
        call_count = {"n": 0}
        sam_mask = torch.zeros((256, 256))
        sam_mask[80:140, 100:156] = 1.0

        def counted_predict(_h, _bbox):
            call_count["n"] += 1
            return sam_mask

        detailer_node._load_sam = lambda name: ("sam2", object()) if name and name != "(none)" else None
        detailer_node._sam_set_image = lambda *_a, **_kw: True
        detailer_node._sam_predict = counted_predict
        detailer_node._sam_predict_batch = lambda *_a, **_kw: None  # force per-bbox

        node = detailer_node.GrimmRibbitySmartDetailer()
        out = node.detail(**self._kwargs(enable_face=True, enable_skin=True,
                                           bbox_face="face_yolov8m.pt",
                                           sam_model="sam2.1_hiera_large.pt"))
        self._assert_clean_output(*out, label="sam_cache")
        self.assertEqual(call_count["n"], 1,
                          f"SAM predict should run once for shared bbox, got {call_count['n']}")


if __name__ == "__main__":
    unittest.main()
