"""GrimmRibbity Smart Detailer.

Self-contained one-node face/eyes/hands/skin detailer. No Comfy custom-node
packs required — uses ultralytics for YOLO, optionally sam2/segment_anything
for mask refinement, and ComfyUI's built-in `common_ksampler` + VAE.

Pipeline per target (face → skin → eyes → hands):
  1. YOLO bbox detection — cached per detector model file
  2. Optional SAM mask refinement per bbox — tighter, follows the actual shape
  3. Crop with `crop_factor`, upscale to `guide_size` (multiple of 8)
  4. Encode per-target wildcard text once, ConditioningConcat onto positive
  5. VAE-encode crop, attach noise_mask, run common_ksampler
  6. VAE-decode (tiled by default for OOM safety)
  7. Composite back: blend along the SAM mask (or feathered bbox if no SAM)
  8. soft_empty_cache between targets to keep host RAM tight on long batches

Optimisations baked in:
  - Detection cache: when two targets share a YOLO detector + threshold, the
    second pass reuses the first pass's bboxes (no second YOLO inference).
  - Wildcard cache: per-target CLIPTextEncode runs once per target, not once
    per detected bbox.
  - Per-target overrides: face_/eyes_/hands_/skin_threshold and _denoise floats
    let you tune each pass without touching the global defaults. -1 = global.
  - max_per_target: cap detections (1 for portrait-only, 0 for unlimited).

Outputs:
  - image: detailed image
  - mask: union of all per-target detail masks
  - detections_preview: input image with colored bboxes per target (face=green,
    skin=yellow, eyes=cyan, hands=magenta) — wire to a SaveImage to debug.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import folder_paths
import torch
import torch.nn.functional as F


# Hot-path imports hoisted from inside loops/helpers to module load. These
# all resolve through ComfyUI core modules that are guaranteed loaded by the
# time our INPUT_TYPES is queried — no need for the lazy-import dance the
# Impact Pack wrapper used to do.
_NODES_MOD: Any = None
_CLIP_TEXT_ENCODE = None
_COND_CONCAT = None
_COMMON_KSAMPLER = None
_PROGRESS_BAR_CLS = None


def _resolve_comfy_helpers() -> None:
    """Resolve the comfy/nodes singletons we use in hot paths. Called once
    lazily because at module-import time ComfyUI's `nodes` module isn't
    fully populated yet (load order)."""
    global _NODES_MOD, _CLIP_TEXT_ENCODE, _COND_CONCAT, _COMMON_KSAMPLER, _PROGRESS_BAR_CLS
    if _COMMON_KSAMPLER is not None:
        return
    try:
        import nodes as _n
        _NODES_MOD = _n
        _COMMON_KSAMPLER = _n.common_ksampler
        _CLIP_TEXT_ENCODE = _n.CLIPTextEncode()
        _COND_CONCAT = _n.ConditioningConcat()
    except Exception as exc:
        logging.warning("[SmartDetailer] couldn't preload comfy helpers: %s", exc)
    try:
        from comfy.utils import ProgressBar
        _PROGRESS_BAR_CLS = ProgressBar
    except Exception:
        _PROGRESS_BAR_CLS = None


_TARGET_ORDER = ("face", "skin", "eyes", "hands")

_PRESETS: dict[str, dict[str, Any]] = {
    "face":  {"denoise": 0.40, "feather": 12, "crop_factor": 3.0,
              "wildcard": "detailed face, sharp features,",
              "color": (0.30, 0.95, 0.40)},   # green
    "skin":  {"denoise": 0.30, "feather": 20, "crop_factor": 2.5,
              "wildcard": "smooth skin texture, natural pores,",
              "color": (0.95, 0.90, 0.30)},   # yellow
    "eyes":  {"denoise": 0.40, "feather": 15, "crop_factor": 1.5,
              "wildcard": "detailed eyes, highly detailed,",
              "color": (0.30, 0.85, 0.95)},   # cyan
    "hands": {"denoise": 0.45, "feather": 10, "crop_factor": 2.0,
              "wildcard": "detailed hands, anatomically correct fingers,",
              "color": (0.95, 0.30, 0.85)},   # magenta
}

_NONE = "(none)"

# SAM2 weight name -> bundled config yaml. Mirrors Impact Pack's table so the
# files the user already downloaded for Impact still load via our own path.
_SAM2_CONFIGS = {
    "sam2.1_hiera_large.pt":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small.pt":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny.pt":      "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_hiera_large.pt":       "configs/sam2/sam2_hiera_l.yaml",
    "sam2_hiera_base_plus.pt":   "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_small.pt":       "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_tiny.pt":        "configs/sam2/sam2_hiera_t.yaml",
}


def _register_folder_paths() -> None:
    """Idempotent: register ultralytics_bbox/segm/sams folders so dropdowns
    populate even without Impact Pack/Subpack installed."""
    base = folder_paths.models_dir
    exts = getattr(folder_paths, "supported_pt_extensions",
                   {".pt", ".pth", ".safetensors"})
    targets = (
        ("ultralytics_bbox", os.path.join(base, "ultralytics", "bbox")),
        ("ultralytics_segm", os.path.join(base, "ultralytics", "segm")),
        ("ultralytics",      os.path.join(base, "ultralytics")),
    )
    for key, path in targets:
        if key not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[key] = ([path], exts)
        elif path not in folder_paths.folder_names_and_paths[key][0]:
            folder_paths.folder_names_and_paths[key][0].append(path)


_register_folder_paths()


def _list_with_none(folder_key: str) -> list[str]:
    try:
        files = folder_paths.get_filename_list(folder_key)
    except Exception:
        files = []
    return [_NONE] + list(files)


_YOLO_CACHE: dict[str, Any] = {}
_SAM_CACHE: dict[str, tuple[str, Any]] = {}


def _resolve_bbox_path(model_name: str) -> str | None:
    if not model_name or model_name == _NONE:
        return None
    raw = model_name
    if raw.startswith("bbox/"):
        path = folder_paths.get_full_path("ultralytics_bbox", raw[5:])
        if path:
            return path
    elif raw.startswith("segm/"):
        path = folder_paths.get_full_path("ultralytics_segm", raw[5:])
        if path:
            return path
    for key in ("ultralytics_bbox", "ultralytics_segm", "ultralytics"):
        path = folder_paths.get_full_path(key, raw)
        if path:
            return path
    return None


def _load_yolo(model_name: str):
    if not model_name or model_name == _NONE:
        return None
    cached = _YOLO_CACHE.get(model_name)
    if cached is not None:
        return cached
    path = _resolve_bbox_path(model_name)
    if path is None:
        raise FileNotFoundError(
            f"GrimmRibbity Smart Detailer: YOLO model '{model_name}' not found "
            f"in ultralytics_bbox / ultralytics_segm / ultralytics folders.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "GrimmRibbity Smart Detailer requires the 'ultralytics' package. "
            "Install with: `pip install ultralytics` in your ComfyUI venv.") from exc
    yolo = YOLO(path)
    # Pin to GPU when available so first inference doesn't pay the move cost,
    # and fuse Conv+BN layers (~10-20% inference speedup, free since we cache
    # the loaded model anyway). Both calls are safe no-ops on failure.
    try:
        if torch.cuda.is_available():
            yolo.to("cuda")
    except Exception as exc:
        logging.warning("[SmartDetailer] YOLO GPU pin failed (%s) — CPU inference.", exc)
    try:
        if hasattr(yolo, "fuse"):
            yolo.fuse()
    except Exception as exc:
        logging.warning("[SmartDetailer] YOLO fuse() failed (%s) — using unfused.", exc)
    _YOLO_CACHE[model_name] = yolo
    return yolo


def _load_sam(model_name: str):
    """Returns (kind, predictor) where kind is 'sam2' or 'sam_v1', or None for
    the (none) sentinel. Cached for the process lifetime."""
    if not model_name or model_name == _NONE:
        return None
    cached = _SAM_CACHE.get(model_name)
    if cached is not None:
        return cached

    path = folder_paths.get_full_path("sams", model_name)
    if path is None:
        raise FileNotFoundError(
            f"GrimmRibbity Smart Detailer: SAM model '{model_name}' not found "
            f"in the 'sams' folder.")

    if model_name in _SAM2_CONFIGS:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 requested but the 'sam2' package isn't installed. "
                "`pip install sam2` in your ComfyUI venv, or pick a SAM v1 "
                "checkpoint instead (sam_vit_*.pth).") from exc
        sam = build_sam2(_SAM2_CONFIGS[model_name], path)
        handle = ("sam2", SAM2ImagePredictor(sam))
    else:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM v1 requested but 'segment_anything' isn't installed. "
                "`pip install segment_anything` in your ComfyUI venv, or use "
                "a SAM2 checkpoint (sam2.1_hiera_*.pt).") from exc
        kind = next((k for k in ("vit_h", "vit_l", "vit_b") if k in model_name), "vit_b")
        sam = sam_model_registry[kind](checkpoint=path)
        handle = ("sam_v1", SamPredictor(sam))

    _SAM_CACHE[model_name] = handle
    return handle


def _sam_set_image(sam_handle, image_hwc_uint8) -> bool:
    """Pin the image into the SAM predictor once. Subsequent _sam_predict
    calls reuse the cached embeddings — set_image runs the SAM encoder
    (~30-50 ms on sam2.1_hiera_large) so hoisting it from per-bbox to
    per-target saves real cycles when the same image has multiple
    detections. Returns True on success."""
    if sam_handle is None:
        return False
    try:
        sam_handle[1].set_image(image_hwc_uint8)
        return True
    except Exception as exc:
        logging.warning("[SmartDetailer] SAM set_image failed (%s) — "
                        "falling back to feathered bbox.", exc)
        return False


def _sam_predict(sam_handle, bbox: tuple[int, int, int, int]
                  ) -> torch.Tensor | None:
    """Run SAM predict() on the already-set image with the bbox as prompt.
    Returns a (H, W) float tensor mask, or None on failure."""
    import numpy as np
    if sam_handle is None:
        return None
    try:
        _kind, predictor = sam_handle
        box = np.array([list(bbox)], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        if masks.ndim == 4:
            masks = masks[0]
            scores = scores[0] if scores.ndim == 2 else scores
        best = int(np.argmax(scores))
        return torch.from_numpy(masks[best].astype("float32"))
    except Exception as exc:
        logging.warning("[SmartDetailer] SAM predict failed (%s) — "
                        "falling back to feathered bbox.", exc)
        return None


def _sam_predict_batch(
    sam_handle, bboxes: list[tuple[int, int, int, int]],
) -> list[torch.Tensor | None] | None:
    """Run SAM predict() once for all bboxes in a target. Both SAM2 (native
    batched prompts via box=(N,4)) and SAM v1 (predict_torch with transformed
    boxes) parallelise — for 3+ bboxes this saves N-1 SAM forwards. Returns
    a list aligned with `bboxes` (None for any failed bbox), or None on
    total batch failure (caller falls back to per-bbox loop)."""
    import numpy as np
    if sam_handle is None or not bboxes:
        return None
    kind, predictor = sam_handle
    try:
        if kind == "sam2":
            box_array = np.array([list(b) for b in bboxes], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=box_array, multimask_output=True)
            # Expected: (N, M, H, W) for masks. Some SAM2 versions return
            # (N, H, W) when M is squeezed — handle both.
            out: list[torch.Tensor | None] = []
            if masks.ndim == 3:
                for i in range(masks.shape[0]):
                    out.append(torch.from_numpy(masks[i].astype("float32")))
            else:
                for i in range(masks.shape[0]):
                    row_scores = scores[i] if scores.ndim >= 2 else scores
                    best = int(np.argmax(row_scores))
                    out.append(torch.from_numpy(masks[i, best].astype("float32")))
            return out

        # SAM v1: predict_torch path — needs transformed boxes on the
        # predictor's device. apply_boxes_torch handles the resize from
        # input image coords to model input coords.
        box_tensor = torch.tensor([list(b) for b in bboxes], dtype=torch.float32,
                                    device=predictor.device)
        transformed = predictor.transform.apply_boxes_torch(
            box_tensor, predictor.original_size)
        masks_t, scores_t, _ = predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=transformed, multimask_output=True)
        # masks_t shape: (N, M, H, W); scores_t: (N, M)
        out_v1: list[torch.Tensor | None] = []
        for i in range(masks_t.shape[0]):
            best = int(scores_t[i].argmax().item())
            out_v1.append(masks_t[i, best].to(torch.float32).cpu())
        return out_v1
    except Exception as exc:
        logging.warning("[SmartDetailer] SAM batched predict failed (%s) — "
                        "falling back to per-bbox predict.", exc)
        return None


def _detect_bboxes_with_conf(yolo, image_hwc_uint8, threshold: float,
                              drop_size: int, max_n: int = 0,
                              nms_iou: float = 0.5, imgsz: int = 640,
                              ) -> list[tuple[tuple[int, int, int, int], float]]:
    """Run YOLO. Returns sorted-by-conf-desc list of ((x1,y1,x2,y2), conf).

    `max_n=0` keeps all; otherwise keeps top N by confidence.
    `nms_iou` controls how aggressively overlapping detections are merged
    (lower = stricter merge, fewer duplicates).
    `imgsz` is YOLO's inference resolution — higher catches smaller faces
    at proportional cost; 640 is the YOLO default."""
    results = yolo(image_hwc_uint8, conf=threshold, iou=nms_iou,
                    imgsz=imgsz, verbose=False)
    detections: list[tuple[tuple[int, int, int, int], float]] = []
    for result in results:
        if result.boxes is None or result.boxes.xyxy is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None \
                else [0.0] * len(xyxy)
        for row, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = (int(v) for v in row)
            if (x2 - x1) < drop_size or (y2 - y1) < drop_size:
                continue
            detections.append(((x1, y1, x2, y2), float(conf)))
    detections.sort(key=lambda d: d[1], reverse=True)
    if max_n and max_n > 0:
        detections = detections[:max_n]
    return detections


def _expand_bbox(bbox, crop_factor: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * crop_factor, (y2 - y1) * crop_factor
    nx1, ny1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    nx2, ny2 = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
    return nx1, ny1, nx2, ny2


def _scale_to_guide(crop_h: int, crop_w: int, guide: int, max_size: int
                    ) -> tuple[int, int]:
    short = min(crop_h, crop_w)
    if short < 1:
        return crop_h, crop_w
    scale = guide / short
    new_h, new_w = int(crop_h * scale), int(crop_w * scale)
    if max(new_h, new_w) > max_size:
        scale *= max_size / max(new_h, new_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)
    return max(64, (new_h // 8) * 8), max(64, (new_w // 8) * 8)


def _make_feather_mask(h: int, w: int, feather: int, device, dtype=torch.float32,
                        suppress_edges: tuple[bool, bool, bool, bool] = (False, False, False, False),
                        ) -> torch.Tensor:
    """Rectangular fall-off mask. Used as fallback when SAM isn't wired.

    `suppress_edges` is (left, top, right, bottom): when an edge of the bbox
    abuts the actual image border, the feather on THAT side is replaced with
    a flat 1.0 — without this, the mask drops to 0 at the image edge and
    you get a visible seam where the original (un-detailed) edge meets the
    feathered (zero-weighted) detail. Edge-aware feathering eliminates that.
    """
    if feather <= 0:
        return torch.ones((1, h, w), device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    ys = torch.arange(h, device=device, dtype=dtype)
    sup_l, sup_t, sup_r, sup_b = suppress_edges
    # Distance to the nearer x-edge, suppressed where the bbox touches a real
    # image border (so the feather doesn't dip to 0 there).
    dist_l = xs if not sup_l else torch.full_like(xs, feather)
    dist_r = (w - 1) - xs if not sup_r else torch.full_like(xs, feather)
    edge_x = torch.minimum(dist_l, dist_r).clamp(max=feather) / feather
    dist_t = ys if not sup_t else torch.full_like(ys, feather)
    dist_b = (h - 1) - ys if not sup_b else torch.full_like(ys, feather)
    edge_y = torch.minimum(dist_t, dist_b).clamp(max=feather) / feather
    return (edge_y[:, None] * edge_x[None, :]).unsqueeze(0)


def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Pure-torch separable gaussian. x: (N, 1, H, W) → same shape, blurred."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kx = kernel_1d.view(1, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, radius))
    x = F.conv2d(x, ky, padding=(radius, 0))
    return x


def _refine_to_blend_mask(
    sam_mask: torch.Tensor | None,   # (h_crop, w_crop) or full-res
    crop_box: tuple[int, int, int, int],
    crop_h: int, crop_w: int,
    feather: int, device, dtype,
    image_h: int, image_w: int,
) -> torch.Tensor:
    """Build the (1, crop_h, crop_w) alpha mask used to composite a refined
    crop back. With SAM: gaussian-blur the SAM mask so the seam follows the
    actual region outline. Without: feathered bbox with edge-suppression
    where the bbox touches the real image border."""
    x1, y1, x2, y2 = crop_box
    if sam_mask is None:
        suppress = (x1 <= 0, y1 <= 0, x2 >= image_w, y2 >= image_h)
        return _make_feather_mask(
            crop_h, crop_w, feather, device=device, dtype=dtype,
            suppress_edges=suppress,
        )

    crop_mask = sam_mask[y1:y2, x1:x2].to(device=device, dtype=dtype)
    crop_mask = crop_mask.unsqueeze(0).unsqueeze(0)
    if crop_mask.shape[-2:] != (crop_h, crop_w):
        crop_mask = F.interpolate(crop_mask, size=(crop_h, crop_w),
                                   mode="bilinear", align_corners=False)
    blurred = _gaussian_blur_2d(crop_mask, sigma=max(1.0, feather / 2.0))
    return blurred.squeeze(0).clamp(0, 1)


# Module-level wildcard CLIP encoding cache. Survives across detail() calls
# so a batched workflow that reuses the same wildcard + checkpoint encodes
# ONCE total, not once per queue. Bounded LRU to cap memory — keys are
# (id(clip), wildcard_text); values are the encoded conditioning tensor.
_WILDCARD_CACHE: OrderedDict[tuple[int, str], Any] = OrderedDict()
_WILDCARD_CACHE_MAX = 32


def _encode_wildcard_cached(clip, wildcard_text: str, positive):
    """ConditioningConcat the wildcard onto positive. The encoded wildcard
    tensor is cached at module level by (id(clip), text) — same prompt + same
    CLIP runs CLIPTextEncode once total, not once per detected bbox or per
    queued image."""
    if not wildcard_text or not wildcard_text.strip():
        return positive
    if _CLIP_TEXT_ENCODE is None or _COND_CONCAT is None:
        _resolve_comfy_helpers()
        if _CLIP_TEXT_ENCODE is None:  # still missing
            return positive
    key = (id(clip), wildcard_text)
    wc_cond = _WILDCARD_CACHE.get(key)
    if wc_cond is None:
        (wc_cond,) = _CLIP_TEXT_ENCODE.encode(clip, wildcard_text)
        _WILDCARD_CACHE[key] = wc_cond
        if len(_WILDCARD_CACHE) > _WILDCARD_CACHE_MAX:
            _WILDCARD_CACHE.popitem(last=False)
    else:
        # Touch for LRU: move-to-end so cold entries get evicted first.
        _WILDCARD_CACHE.move_to_end(key)
    (combined,) = _COND_CONCAT.concat(positive, wc_cond)
    return combined


def _vae_decode(vae, samples_dict, tiled: bool):
    """Decode latent → image. When `tiled` is requested, we only ACTUALLY tile
    if the decoded resolution would exceed ~1024² — for typical face/eye
    crops (256-512px decoded) full decode is faster and plenty RAM-safe.
    Tiling kicks in for hands/skin passes at higher guide_size."""
    samples = samples_dict["samples"]
    # latent shape (B, C, H, W); decoded image is 8x in H+W.
    h_pix, w_pix = samples.shape[-2] * 8, samples.shape[-1] * 8
    needs_tile = tiled and (h_pix > 1024 or w_pix > 1024)
    if needs_tile:
        try:
            return vae.decode_tiled(samples, tile_x=512, tile_y=512)
        except TypeError:
            return vae.decode_tiled(samples)
    return vae.decode(samples)


def _draw_bbox_on_preview(preview: torch.Tensor, bbox, color, line: int = 2,
                            ) -> None:
    """In-place: stroke a rectangle on a (1, H, W, 3) image tensor."""
    _, H, W, _ = preview.shape
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    rgb = torch.tensor(color, device=preview.device, dtype=preview.dtype)
    # top + bottom strips
    preview[:, max(0, y1):min(H, y1 + line), x1:x2, :] = rgb
    preview[:, max(0, y2 - line):y2, x1:x2, :] = rgb
    # left + right strips
    preview[:, y1:y2, x1:min(W, x1 + line), :] = rgb
    preview[:, y1:y2, max(0, x2 - line):x2, :] = rgb


# Tiny 5x7 bitmap font for preview bbox labels — pure torch, no PIL/cv2 dep.
# Each glyph is a (7, 5) row of bits; row[y]&(1<<(4-x)) is the pixel.
# Only the chars we need: target labels (face/eyes/hands/skin) + digits + "."
_FONT_5x7 = {
    "f": [0x07, 0x08, 0x1E, 0x08, 0x08, 0x08, 0x08],
    "a": [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x11, 0x11],
    "c": [0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E],
    "e": [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0F],
    "y": [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
    "s": [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
    "h": [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
    "n": [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
    "d": [0x01, 0x01, 0x0D, 0x13, 0x11, 0x13, 0x0D],
    "k": [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
    "i": [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
    " ": [0x00] * 7,
    ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06],
    "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "2": [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
    "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
    "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
}


def _draw_text_on_preview(preview: torch.Tensor, x: int, y: int, text: str,
                            color, scale: int = 1) -> None:
    """In-place: render `text` on (1, H, W, 3) tensor at (x, y) using the
    5x7 bitmap font. Each pixel becomes a `scale x scale` block. Off-canvas
    pixels are clipped silently."""
    _, H, W, _ = preview.shape
    rgb = torch.tensor(color, device=preview.device, dtype=preview.dtype)
    char_w, char_h = 5, 7
    glyph_w = char_w * scale + scale  # +scale for inter-char spacing
    cx = x
    for ch in text.lower():
        glyph = _FONT_5x7.get(ch)
        if glyph is None:
            cx += glyph_w
            continue
        for gy in range(char_h):
            row_bits = glyph[gy]
            for gx in range(char_w):
                if row_bits & (1 << (char_w - 1 - gx)):
                    py0 = y + gy * scale
                    px0 = cx + gx * scale
                    py1 = min(H, py0 + scale)
                    px1 = min(W, px0 + scale)
                    if py0 >= 0 and px0 >= 0 and py1 > py0 and px1 > px0:
                        preview[:, py0:py1, px0:px1, :] = rgb
        cx += glyph_w


@dataclass
class _Pass:
    name: str
    yolo: Any
    yolo_key: str
    threshold: float
    denoise: float
    max_n: int
    steps: int = 0  # 0 = use the global steps value
    detections: list[tuple[tuple[int, int, int, int], float]] = field(default_factory=list)


def _vae_encode(vae, pixels, tiled: bool):
    """Encode pixels → latent. Tiled mode kicks in for large crops to keep
    encode RAM bounded — symmetric with _vae_decode's auto-tile policy."""
    h, w = pixels.shape[1], pixels.shape[2]
    if tiled and (h > 1024 or w > 1024):
        try:
            return vae.encode_tiled(pixels, tile_x=512, tile_y=512)
        except (TypeError, AttributeError):
            return vae.encode(pixels)
    return vae.encode(pixels)


def _enhance_one_pass(
    image: torch.Tensor, model, clip, vae, positive, negative,
    plan: _Pass, sam_handle, *,
    img_uint8_np,                # precomputed once per detail() call
    seed: int, steps: int, cfg: float,
    sampler_name: str, scheduler: str,
    guide_size: int, max_size: int,
    drop_size: int, force_inpaint: bool,
    tiled_decode: bool, tiled_encode: bool,
    mask_strength: float, same_seed: bool,
    wildcard_text: str,
    sam_mask_cache: dict[tuple[int, int, int, int], torch.Tensor | None],
    soft_cleanup: Callable[[], None] | None,
    progress_bar=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _COMMON_KSAMPLER is None:
        _resolve_comfy_helpers()
    common_ksampler = _COMMON_KSAMPLER

    device = image.device
    _, H, W, _ = image.shape

    if not plan.detections:
        return image, torch.zeros((1, H, W), device=device, dtype=image.dtype)

    feather = int(_PRESETS[plan.name]["feather"])
    crop_factor = float(_PRESETS[plan.name]["crop_factor"])

    positive_with_wc = _encode_wildcard_cached(clip, wildcard_text, positive)
    running = image.clone()
    combined_mask = torch.zeros((1, H, W), device=device, dtype=image.dtype)

    # SAM's set_image runs the SAM encoder once per target. The encoder
    # output is cached internally by the predictor — we only need to call
    # set_image when the input image changes (it doesn't between bboxes).
    sam_ready = _sam_set_image(sam_handle, img_uint8_np) if sam_handle else False

    # Identify bboxes whose SAM mask hasn't been computed yet across the
    # whole detail() call. Face + skin passes share bboxes (same detector +
    # threshold) — SAM result is identical, so we cache by raw_bbox tuple.
    needs_sam = []
    if sam_ready:
        for raw_bbox, _conf in plan.detections:
            if raw_bbox not in sam_mask_cache:
                needs_sam.append(raw_bbox)

    # If 2+ uncached bboxes via SAM2, batched predict: ~Nx faster than the
    # per-bbox loop. Falls back to per-bbox when batching returns None
    # (SAM v1, batch failure).
    if sam_ready and len(needs_sam) > 1:
        batch_results = _sam_predict_batch(sam_handle, needs_sam)
        if batch_results is not None:
            for bbox, mask in zip(needs_sam, batch_results):
                sam_mask_cache[bbox] = mask
        else:
            for bbox in needs_sam:
                sam_mask_cache[bbox] = _sam_predict(sam_handle, bbox)
    elif sam_ready and len(needs_sam) == 1:
        sam_mask_cache[needs_sam[0]] = _sam_predict(sam_handle, needs_sam[0])

    for i, (raw_bbox, _conf) in enumerate(plan.detections):
        # Soft-cancel: between every bbox, check if user hit the "Interrupt"
        # button. Comfy raises an InterruptProcessingException which propagates
        # through the queue executor and aborts the whole prompt cleanly.
        try:
            import comfy.model_management as _mm
            _mm.throw_exception_if_processing_interrupted()
        except (ImportError, AttributeError):
            pass

        x1, y1, x2, y2 = _expand_bbox(raw_bbox, crop_factor, W, H)
        if x2 - x1 < 16 or y2 - y1 < 16:
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        sam_full_mask = sam_mask_cache.get(raw_bbox) if sam_ready else None

        crop = running[:, y1:y2, x1:x2, :]
        crop_h, crop_w = crop.shape[1], crop.shape[2]
        target_h, target_w = _scale_to_guide(crop_h, crop_w, guide_size, max_size)
        if not force_inpaint and target_h <= crop_h and target_w <= crop_w:
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        upscaled = F.interpolate(
            crop.permute(0, 3, 1, 2), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()

        latent = {"samples": _vae_encode(vae, upscaled[:, :, :, :3], tiled_encode)}
        latent_h, latent_w = latent["samples"].shape[-2:]
        # Latent-space noise mask: respects bbox-edge suppression so the
        # sample pass also matches the eventual composite blend.
        suppress = (x1 <= 0, y1 <= 0, x2 >= W, y2 >= H)
        latent_mask = _make_feather_mask(
            target_h, target_w, feather, device=device,
            dtype=latent["samples"].dtype, suppress_edges=suppress)
        latent_mask = F.interpolate(
            latent_mask.unsqueeze(0), size=(latent_h, latent_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        latent["noise_mask"] = latent_mask

        bbox_seed = int(seed) if same_seed else int(seed) + i
        # Per-target steps override falls back to the function's `steps` arg
        # (which is always the global value at the call site).
        pass_steps = plan.steps if plan.steps > 0 else int(steps)
        (refined_latent,) = common_ksampler(
            model, bbox_seed, pass_steps, float(cfg),
            sampler_name, scheduler,
            positive_with_wc, negative, latent,
            denoise=float(plan.denoise),
        )
        refined_image = _vae_decode(vae, refined_latent, tiled_decode)
        refined_image = refined_image.to(device=device, dtype=running.dtype)
        # nan_to_num BEFORE clamp — clamp(NaN, 0, 1) returns NaN and then casts
        # silently turn it into garbage downstream (the black-screen bug).
        refined_image = torch.nan_to_num(refined_image, nan=0.0,
                                          posinf=1.0, neginf=0.0).clamp(0, 1)

        if refined_image.shape[1] != crop_h or refined_image.shape[2] != crop_w:
            refined_image = F.interpolate(
                refined_image.permute(0, 3, 1, 2), size=(crop_h, crop_w),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()

        composite_mask = _refine_to_blend_mask(
            sam_full_mask, (x1, y1, x2, y2), crop_h, crop_w, feather,
            device=device, dtype=running.dtype,
            image_h=H, image_w=W,
        ).unsqueeze(-1)
        if mask_strength != 1.0:
            composite_mask = composite_mask * float(mask_strength)
        composite_mask = torch.nan_to_num(composite_mask, nan=0.0).clamp(0, 1)
        blended = refined_image * composite_mask + crop * (1 - composite_mask)
        running[:, y1:y2, x1:x2, :] = torch.nan_to_num(blended, nan=0.0).clamp(0, 1)
        combined_mask[:, y1:y2, x1:x2] = torch.maximum(
            combined_mask[:, y1:y2, x1:x2], composite_mask.squeeze(-1),
        )
        if progress_bar is not None:
            progress_bar.update(1)

    if soft_cleanup is not None:
        soft_cleanup()

    return running, combined_mask


class GrimmRibbitySmartDetailer:
    """One-node face/eyes/hands/skin detailer with SAM mask refinement."""

    DESCRIPTION = (
        "Replaces the 3-node FaceDetailer chain. Self-contained: ultralytics "
        "for detection, optional SAM for tighter masks, ComfyUI's built-in "
        "sampler — no Impact Pack, no Subpack. Tiled VAE decode default-on; "
        "per-target threshold/denoise/max overrides; detection preview output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        _register_folder_paths()
        bbox_models = _list_with_none("ultralytics_bbox")
        sam_models = _list_with_none("sams")
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["dpmpp_3m_sde_gpu"]
            schedulers = ["karras"]

        def thr(target):
            return ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                "tooltip": (
                    f"Per-target YOLO confidence threshold for the {target} pass. "
                    f"-1 = use the global bbox_threshold. "
                    f"Higher (0.6-0.8) = stricter, only obvious detections. "
                    f"Lower (0.2-0.4) = catches more, including partial / "
                    f"closed-eye / occluded cases at the cost of false positives. "
                    f"Eye detectors usually want lower thresholds than face detectors.")})

        def den(target, hint):
            return ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                "tooltip": (
                    f"Per-target denoise for the {target} pass. "
                    f"-1 = use global denoise + preset offset (this target's preset is {hint}). "
                    f"Higher (0.5-0.7) = more redrawing, can change identity / lose likeness. "
                    f"Lower (0.2-0.3) = subtle refinement only, good for already-decent regions. "
                    f"Sweet spot is usually 0.35-0.45.")})

        def maxn(target):
            return ("INT", {"default": 0, "min": 0, "max": 64,
                "tooltip": (
                    f"Cap detections for the {target} pass. "
                    f"0 = no cap (every YOLO hit gets detailed). "
                    f"1 = process only the highest-confidence detection (portrait mode — "
                    f"good when you only want the main subject's {target} fixed and don't "
                    f"care about background figures). "
                    f"N = top-N by confidence. Higher = more sampling work.")})

        def stp(target):
            return ("INT", {"default": 0, "min": 0, "max": 200,
                "tooltip": (
                    f"Per-target sample steps for the {target} pass. "
                    f"0 = use the global steps value. "
                    f"Useful for cheap eyes/skin passes (10-15 steps) paired with a "
                    f"thorough face pass (20-30 steps), or vice versa. Cost scales linearly.")})

        return {
            "required": {
                "image": ("IMAGE", {"tooltip":
                    "Image(s) to detail. Wire from your sampler's IMAGE output. Batch input "
                    "is supported but each image is processed independently — for video, use a "
                    "video-aware detailer instead."}),
                "model": ("MODEL", {"tooltip":
                    "MODEL used for the inpainting sample passes. Should be the same checkpoint "
                    "(or a tuned variant) as the one that produced the input image — wildly "
                    "different models will fight over the cropped region's identity."}),
                "clip": ("CLIP", {"tooltip":
                    "CLIP from the same checkpoint as MODEL. Used to encode each target's "
                    "wildcard prompt (e.g. 'detailed eyes, highly detailed') so the sample pass "
                    "knows what to look for."}),
                "vae": ("VAE", {"tooltip":
                    "VAE used to encode the cropped region into latent space (before sampling) "
                    "and decode the result back. Tile_decode below controls whether decode "
                    "splits the latent into tiles (RAM-saver)."}),
                "positive": ("CONDITIONING", {"tooltip":
                    "Your full positive prompt conditioning. Each target's preset wildcard is "
                    "ConditioningConcat'd onto this — your subject prompt + 'detailed face, "
                    "sharp features,' for the face pass, etc. Encoded once per target, reused "
                    "across all detected bboxes in that target."}),
                "negative": ("CONDITIONING", {"tooltip":
                    "Your full negative prompt conditioning. Used unchanged on every target's "
                    "sample pass — same negative as your main gen is the right starting point."}),

                "enable_face":  ("BOOLEAN", {"default": True, "tooltip":
                    "Run the face pass: detect faces with bbox_face, sample with 'detailed face, "
                    "sharp features,' wildcard at preset denoise 0.40, crop_factor 3.0 (lots of "
                    "context around the face). Most useful first pass."}),
                "enable_eyes":  ("BOOLEAN", {"default": True, "tooltip":
                    "Run the eyes pass: tight crop (crop_factor 1.5) using bbox_eyes if set, "
                    "else bbox_face. Wildcard 'detailed eyes, highly detailed,'. Best run AFTER "
                    "face — sharpens irises and pupils on the freshly-detailed face."}),
                "enable_hands": ("BOOLEAN", {"default": False, "tooltip":
                    "Run the hands pass: needs a dedicated hand detector (hand_yolov8s.pt). "
                    "Highest preset denoise (0.45) since hands often need real geometry "
                    "rebuilding. Off by default because most workflows don't have a hand model "
                    "wired in."}),
                "enable_skin":  ("BOOLEAN", {"default": False, "tooltip":
                    "Run a skin-smoothing pass on the face region (uses bbox_face). Lowest "
                    "denoise (0.30) and largest feather — refines pore texture without changing "
                    "identity. Off by default; turn on for portrait close-ups."}),

                "bbox_face":  (bbox_models, {"default": _NONE, "tooltip":
                    "YOLO bbox model for the face / skin / (fallback) eyes passes. Pick "
                    "face_yolov8m.pt (good speed-accuracy balance) or face_yolov9c.pt (higher "
                    "accuracy, slower). face_yolov8s.pt is the smallest/fastest. Required when "
                    "enable_face / enable_eyes / enable_skin is True."}),
                "bbox_eyes":  (bbox_models, {"default": _NONE, "tooltip":
                    "Optional dedicated eye detector (e.g. Eyeful_v2-Paired.pt). "
                    "Set to (none) to fall back to bbox_face — when both targets use the same "
                    "detector + threshold, YOLO inference runs ONCE and the bboxes are reused. "
                    "A dedicated eye detector finds eye bboxes inside the face region for "
                    "tighter crops than the face detector alone."}),
                "bbox_hands": (bbox_models, {"default": _NONE, "tooltip":
                    "YOLO bbox model for hands (e.g. hand_yolov8s.pt). Required when "
                    "enable_hands is True; ignored otherwise."}),
                "sam_model":  (sam_models, {"default": _NONE, "tooltip":
                    "Optional SAM (Segment Anything) model for mask refinement. Without SAM, "
                    "the composite blends the refined crop back using a rectangular feathered "
                    "mask — visible in some cases as a soft rectangular halo. With SAM, the "
                    "blend follows the actual face / hand outline so the seam is invisible. "
                    "sam2.1_hiera_large.pt is highest quality (~600 MB load, ~15-25% slower per "
                    "bbox). Loaded once, cached for the session."}),

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip":
                    "Seed for the sampling passes. Per-target seed is seed + (target_index * "
                    "1000) so the four passes don't share noise. Keeping this fixed across runs "
                    "gives reproducible detail."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip":
                    "Sample steps per detection per target. 20 is a good baseline. More steps "
                    "= slightly cleaner detail at proportional time cost. With low denoise "
                    "(0.3-0.4) the effective sample budget is steps * denoise, so stepping up "
                    "from 20 to 40 only meaningfully changes things if denoise is also high."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip":
                    "CFG scale for the sample passes. 5-7 is the SDXL detailer sweet spot; "
                    "higher than your main gen's CFG often over-bakes the face. Reduce to 2-4 "
                    "if using Turbo / Lightning / Hyper checkpoints (low-CFG models)."}),
                "sampler_name": (samplers, {
                    "default": "dpmpp_3m_sde_gpu" if "dpmpp_3m_sde_gpu" in samplers else samplers[0],
                    "tooltip":
                    "Sampler used for every detection's sample pass. dpmpp_3m_sde_gpu pairs "
                    "well with the karras scheduler for SDXL detailing. euler_ancestral is a "
                    "lighter alternative. Match your main gen's sampler if results clash."}),
                "scheduler": (schedulers, {
                    "default": "karras" if "karras" in schedulers else schedulers[0],
                    "tooltip":
                    "Scheduler for the sample passes. karras + dpmpp_3m_sde_gpu is the "
                    "go-to combo. exponential / sgm_uniform also work; simple is too coarse "
                    "at 20 steps."}),

                "denoise": ("FLOAT", {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip":
                    "Global denoise applied to the inpainting passes. Per-target presets "
                    "ride on top: face/eyes use this value directly, hands run +0.05, skin runs "
                    "-0.10. Higher = more aggressive redraw (can change identity); lower = more "
                    "conservative (preserves likeness). 0.35-0.45 is the safe range. Per-target "
                    "*_denoise overrides win when set."}),
                "guide_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip":
                    "Target short-edge size (px) the cropped region is upscaled to before "
                    "sampling. 512 = SD1.5-native, 1024 = SDXL-native — match your model's "
                    "training resolution. Smaller = faster but less detail; larger = sharper "
                    "but more VRAM and may exceed your VAE's safe range."}),
                "max_size":   ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip":
                    "Hard cap on the longer-edge after guide-size scaling, to stop very wide "
                    "crops from blowing up. If the upscaled crop would exceed this, the scale "
                    "is reduced so longest edge = max_size. 1024-1536 is sane for SDXL."}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip":
                    "Global YOLO confidence threshold — detections below this score are "
                    "ignored. 0.5 is a balanced default. Drop to 0.3 if a partial/turned face "
                    "isn't being caught; raise to 0.7 if false positives keep showing up. "
                    "Per-target *_threshold overrides win when set (eye detectors often need "
                    "0.3-0.4 to catch closed eyes)."}),
                "max_per_target": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip":
                    "Global cap on detections per target. 0 = unlimited (every YOLO hit gets "
                    "sampled). 1 = portrait mode — only the highest-confidence detection runs, "
                    "skip background faces / extra hands. 2-3 = duo / small group. Higher = "
                    "crowd. Each detection is one sample pass, so this is the main 'how slow "
                    "will this be' knob. Per-target *_max overrides win when > 0."}),

                "tiled_decode": ("BOOLEAN", {"default": True, "tooltip":
                    "Decode the refined latent in 512px tiles when the decoded image would "
                    "exceed 1024². Tiled mode roughly halves peak host RAM during VAE decode "
                    "— directly fixes the OOM seen on long batches. Auto-disabled for small "
                    "crops where full decode is faster. Disable entirely only if you see "
                    "visible tile seams (rare)."}),
                "tiled_encode": ("BOOLEAN", {"default": False, "tooltip":
                    "Encode the upscaled crop in 512px tiles when crop exceeds 1024². OFF "
                    "by default — encode is rarely the OOM bottleneck. Turn ON if you crank "
                    "guide_size to 1280-1536 and start seeing encode OOMs."}),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip":
                    "Multiplier on the composite blend mask. 1.0 = full replacement of the "
                    "detected region. 0.5-0.8 = subtle blend, keeps more of the original "
                    "(useful for the skin pass to soften pore work). 1.2-1.5 = harder edge "
                    "(rare; only when the SAM mask is too soft and detail bleeds outside)."}),
                "same_seed_per_target": ("BOOLEAN", {"default": False, "tooltip":
                    "When True, every detected bbox in a target uses the same seed (seed + "
                    "target_index*1000). Default False = each bbox gets seed + bbox_index, "
                    "giving variety across multiple faces. Turn ON for repeatable A/B "
                    "comparisons or when one detection per target is the norm."}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip":
                    "When True, the node passes the input image through unchanged (and emits "
                    "the input as the preview, plus an empty mask). Use to A/B compare with vs "
                    "without detailing without rewiring the workflow."}),
            },
            "optional": {
                "wildcard_prefix": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Free-text prepended to every target's preset wildcard. e.g. setting "
                    "'masterpiece, intricate detail,' here makes the face pass run "
                    "'masterpiece, intricate detail, detailed face, sharp features,'. Keep it "
                    "general — target-specific words go in the preset."}),
                # Per-target overrides — fine-grained tuning without touching the globals.
                "face_threshold":  thr("face"),
                "face_denoise":    den("face",  "0.40"),
                "face_max":        maxn("face"),
                "face_steps":      stp("face"),
                "eyes_threshold":  thr("eyes"),
                "eyes_denoise":    den("eyes",  "0.40"),
                "eyes_max":        maxn("eyes"),
                "eyes_steps":      stp("eyes"),
                "hands_threshold": thr("hands"),
                "hands_denoise":   den("hands", "0.45 (= global + 0.05)"),
                "hands_max":       maxn("hands"),
                "hands_steps":     stp("hands"),
                "skin_threshold":  thr("skin"),
                "skin_denoise":    den("skin",  "0.30 (= global - 0.10)"),
                "skin_max":        maxn("skin"),
                "skin_steps":      stp("skin"),
                "force_inpaint": ("BOOLEAN", {"default": True, "tooltip":
                    "When True, every detection runs the sample pass even if the bbox is "
                    "already larger than guide_size. When False, large/clean detections are "
                    "skipped — saves time when most subjects are already detailed enough. "
                    "Default ON because Impact-Pack-style workflows assume always-detail."}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 1024, "tooltip":
                    "Minimum bbox edge length (px) to keep. Detections smaller than this in "
                    "either dimension are dropped — filters out tiny faces/hands too small to "
                    "meaningfully detail anyway. Raise to 30-50 to skip background figures."}),
                "nms_iou": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip":
                    "YOLO's non-max suppression IoU threshold for deduplicating overlapping "
                    "detections. Lower (0.3) = stricter merge — when YOLO returns 2-3 nearly-"
                    "overlapping bboxes for the same face, only one survives. Higher (0.7) = "
                    "more lenient — keeps near-duplicates separate, useful for tight close-ups "
                    "where hand/face bboxes naturally overlap."}),
                "yolo_imgsz": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32,
                    "tooltip":
                    "Resolution YOLO downsamples the input to before detection. 640 is the "
                    "model's training resolution and the speed-quality default. Bump to 960 / "
                    "1280 to catch SMALL faces in wide shots (background characters, group "
                    "scenes) at ~2-3x detection cost. Drop to 416 for low-res anime where the "
                    "face takes most of the frame and 640 is wasted budget."}),
                "max_bbox_area_pct": ("FLOAT", {"default": 0.95, "min": 0.10, "max": 1.0, "step": 0.05,
                    "tooltip":
                    "Sanity cap on bbox size as fraction of image area. Detections covering "
                    "more than this much of the frame are discarded as false positives "
                    "(YOLO occasionally returns whole-image bboxes on confused inputs). 0.95 "
                    "= keep almost everything; 0.5 = drop any detection covering more than "
                    "half the image. Lower if you keep getting nonsense whole-frame detections."}),
                "draw_preview": ("BOOLEAN", {"default": True, "tooltip":
                    "When True, the third output (detections_preview) is the input image with "
                    "colored labeled bboxes drawn per detection. When False, the preview output "
                    "is the input image unchanged — saves a clone + bbox/label drawing work. "
                    "Turn OFF when you don't have the preview output wired anywhere."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "detections_preview")
    OUTPUT_TOOLTIPS = (
        "Detailed image. Same shape as input. Falls through to the input image "
        "unchanged when bypass=True or when no detections fired across any "
        "enabled target. NaN-scrubbed at the seam — downstream SaveImage / "
        "Civitai Save / ThumbnailSaver can rely on values being in [0, 1].",
        "Union mask (1, H, W) of every region that was detailed across all "
        "passes — face + eyes + hands + skin, OR'd together. Use this if you "
        "want to apply additional processing (e.g. another sampler pass) only "
        "to the detailed regions.",
        "Input image with detection bounding boxes drawn per target: face = "
        "green, skin = yellow, eyes = cyan, hands = magenta. Bbox color "
        "matches the per-target preset. Wire this to a SaveImage node to "
        "debug what the YOLO models are actually finding without queueing "
        "expensive sample passes.",
    )
    FUNCTION = "detail"
    CATEGORY = "GrimmRibbity/Detailer"

    def detail(self, image, model, clip, vae, positive, negative,
               enable_face, enable_eyes, enable_hands, enable_skin,
               bbox_face, bbox_eyes, bbox_hands, sam_model,
               seed, steps, cfg, sampler_name, scheduler,
               denoise, guide_size, max_size, bbox_threshold, max_per_target,
               tiled_decode, tiled_encode, mask_strength, same_seed_per_target,
               bypass,
               wildcard_prefix="",
               face_threshold=-1.0, face_denoise=-1.0, face_max=0, face_steps=0,
               eyes_threshold=-1.0, eyes_denoise=-1.0, eyes_max=0, eyes_steps=0,
               hands_threshold=-1.0, hands_denoise=-1.0, hands_max=0, hands_steps=0,
               skin_threshold=-1.0, skin_denoise=-1.0, skin_max=0, skin_steps=0,
               force_inpaint=True, drop_size=10,
               nms_iou=0.5, yolo_imgsz=640,
               max_bbox_area_pct=0.95, draw_preview=True):

        device = image.device
        _, H, W, _ = image.shape

        if bypass:
            return (image,
                    torch.zeros((1, H, W), device=device, dtype=image.dtype),
                    image)

        # Build the plan: ordered list of (target, bbox_model_name, threshold,
        # denoise_override, max_override). Order is face → skin → eyes → hands.
        ovr = {
            "face":  (face_threshold,  face_denoise,  face_max,  face_steps),
            "skin":  (skin_threshold,  skin_denoise,  skin_max,  skin_steps),
            "eyes":  (eyes_threshold,  eyes_denoise,  eyes_max,  eyes_steps),
            "hands": (hands_threshold, hands_denoise, hands_max, hands_steps),
        }
        enables = {
            "face":  enable_face,
            "skin":  enable_skin,
            "eyes":  enable_eyes,
            "hands": enable_hands,
        }
        bbox_pick = {
            "face":  bbox_face,
            "skin":  bbox_face,
            "eyes":  bbox_eyes if bbox_eyes != _NONE else bbox_face,
            "hands": bbox_hands,
        }

        passes: list[_Pass] = []
        for name in _TARGET_ORDER:
            if not enables[name]:
                continue
            model_name = bbox_pick[name]
            if model_name == _NONE:
                logging.warning("[SmartDetailer] '%s' enabled but no detector "
                                "configured — skipping.", name)
                continue
            try:
                yolo = _load_yolo(model_name)
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("[SmartDetailer] '%s' detector load failed: %s",
                                name, exc)
                continue
            t_ovr, d_ovr, m_ovr, s_ovr = ovr[name]
            threshold = t_ovr if t_ovr >= 0.0 else float(bbox_threshold)
            preset_d = _PRESETS[name]["denoise"]
            base_d = float(denoise)
            target_denoise = (d_ovr if d_ovr >= 0.0
                                  else max(0.01, min(1.0, base_d + (preset_d - 0.40))))
            max_n = int(m_ovr) if m_ovr > 0 else int(max_per_target)
            target_steps = int(s_ovr) if s_ovr > 0 else int(steps)
            passes.append(_Pass(
                name=name, yolo=yolo, yolo_key=model_name,
                threshold=threshold, denoise=target_denoise, max_n=max_n,
                steps=target_steps,
            ))

        if not passes:
            return (image,
                    torch.zeros((1, H, W), device=device, dtype=image.dtype),
                    image)

        # Detection cache: (yolo_key, threshold, max_n) → detections.
        # Keeps eyes-on-face from re-running the face detector when configured
        # the same way, and keeps face+skin (which share the face detector)
        # from running detection twice.
        # Single shared image-to-uint8 conversion used by YOLO + SAM + preview.
        det_cache: dict[tuple[str, float, int], list] = {}
        img_uint8_np = (image[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        max_area = float(max_bbox_area_pct) * H * W
        for p in passes:
            key = (p.yolo_key, p.threshold, p.max_n)
            if key in det_cache:
                p.detections = det_cache[key]
            else:
                raw = _detect_bboxes_with_conf(
                    p.yolo, img_uint8_np, p.threshold, drop_size,
                    max_n=p.max_n, nms_iou=float(nms_iou), imgsz=int(yolo_imgsz))
                # Drop whole-image false positives — bboxes whose area
                # exceeds the configured fraction of the frame.
                p.detections = [
                    (b, c) for (b, c) in raw
                    if (b[2] - b[0]) * (b[3] - b[1]) <= max_area
                ]
                if len(p.detections) < len(raw):
                    logging.info("[SmartDetailer] %s: dropped %d oversized "
                                  "false-positive(s) (>%.0f%% of image area)",
                                  p.name, len(raw) - len(p.detections),
                                  float(max_bbox_area_pct) * 100)
                det_cache[key] = p.detections

        # Up-front detection summary. Per-target counts + grand total. Lets the
        # user see at a glance what each pass found before sample work starts.
        total_detections = sum(len(p.detections) for p in passes)
        for p in passes:
            confs = ", ".join(f"{c:.2f}" for _, c in p.detections) or "—"
            logging.info("[SmartDetailer] %s: %d detected (conf %s)",
                          p.name, len(p.detections), confs)

        # Empty fast path — every enabled target found zero detections. No
        # SAM load, no preview clone, no sample work. Return input as preview
        # so downstream SaveImage doesn't fight a clone we never wrote into.
        if total_detections == 0:
            logging.info("[SmartDetailer] 0 total detections across %d enabled "
                          "target(s) — passthrough.", len(passes))
            return (image,
                    torch.zeros((1, H, W), device=device, dtype=image.dtype),
                    image)

        # Detections exist — clone for the bbox overlay only when the user
        # actually wants the preview. With draw_preview=False the preview
        # output is the input image unchanged (no clone, no draw work).
        if draw_preview:
            preview = image.clone()
            for p in passes:
                color = _PRESETS[p.name]["color"]
                for raw_bbox, conf in p.detections:
                    _draw_bbox_on_preview(preview, raw_bbox, color)
                    x1, y1, _x2, _y2 = raw_bbox
                    label = f"{p.name} {conf:.2f}"
                    label_y = y1 - 18 if y1 >= 20 else y1 + 4
                    _draw_text_on_preview(preview, x1 + 2, label_y, label,
                                           color=color, scale=2)
        else:
            preview = image

        # Optionally load SAM up front (single load, used across all passes).
        sam_handle = None
        if sam_model and sam_model != _NONE:
            try:
                sam_handle = _load_sam(sam_model)
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("[SmartDetailer] SAM load failed (%s) — "
                                "running with feathered bbox masks.", exc)

        # Soft cache empty between targets — same RAM-saver pattern the SDXL
        # sampler uses between hires iterations.
        try:
            import comfy.model_management as mm
            soft_cleanup = mm.soft_empty_cache
        except Exception:
            soft_cleanup = None

        # Single ProgressBar across the whole detail() call — ticks once per
        # detection-bbox so the user sees "3/8" not target-only counts.
        if _PROGRESS_BAR_CLS is None:
            _resolve_comfy_helpers()
        pbar = _PROGRESS_BAR_CLS(total_detections) if _PROGRESS_BAR_CLS else None

        running = image
        combined_mask = torch.zeros((1, H, W), device=device, dtype=image.dtype)

        import time
        run_start = time.monotonic()

        # SAM mask cache shared across passes — face + skin (which share the
        # same bboxes) compute SAM once total instead of once each.
        sam_mask_cache: dict[tuple[int, int, int, int], torch.Tensor | None] = {}

        for i, p in enumerate(passes):
            preset = _PRESETS[p.name]
            wildcard_text = (f"{wildcard_prefix.strip()} {preset['wildcard']}".strip()
                              if wildcard_prefix.strip() else preset["wildcard"])
            logging.info(
                "[SmartDetailer] target=%s denoise=%.2f crop=%.2f max=%d threshold=%.2f wc='%s'",
                p.name, p.denoise, preset["crop_factor"], p.max_n, p.threshold, wildcard_text)
            pass_start = time.monotonic()
            running, mask = _enhance_one_pass(
                running, model, clip, vae, positive, negative, p, sam_handle,
                img_uint8_np=img_uint8_np,
                seed=int(seed) + i * 1000, steps=int(steps), cfg=float(cfg),
                sampler_name=sampler_name, scheduler=scheduler,
                guide_size=int(guide_size), max_size=int(max_size),
                drop_size=int(drop_size), force_inpaint=bool(force_inpaint),
                tiled_decode=bool(tiled_decode), tiled_encode=bool(tiled_encode),
                mask_strength=float(mask_strength),
                same_seed=bool(same_seed_per_target),
                wildcard_text=wildcard_text,
                sam_mask_cache=sam_mask_cache,
                soft_cleanup=soft_cleanup,
                progress_bar=pbar,
            )
            combined_mask = torch.maximum(combined_mask, mask)
            logging.info("[SmartDetailer] %s pass: %d bbox(es) in %.1fs (steps=%d)",
                          p.name, len(p.detections),
                          time.monotonic() - pass_start, p.steps or int(steps))

        logging.info("[SmartDetailer] all passes done in %.1fs (%d total detections)",
                      time.monotonic() - run_start, total_detections)

        # Final NaN-scrub on the way out — even if the per-pass guards miss
        # something, downstream nodes can't survive a NaN in IMAGE.
        running = torch.nan_to_num(running, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        combined_mask = torch.nan_to_num(combined_mask, nan=0.0).clamp(0, 1)
        return (running, combined_mask, preview)


NODE_CLASS_MAPPINGS = {"GrimmRibbitySmartDetailer": GrimmRibbitySmartDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GrimmRibbitySmartDetailer": "GrimmRibbity — Smart Detailer",
}
