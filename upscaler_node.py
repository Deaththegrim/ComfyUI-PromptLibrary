"""GrimmRibbity Upscale SDXL — self-contained tile diffusion upscaler.

One node that does what the LoadControlNet → ApplyControlNet →
UltimateSDUpscale chain does: pre-upscale (via an upscale model or
Lanczos), apply an SDXL tile ControlNet to the conditioning, then walk
the canvas tile-by-tile running a low-denoise sampler on each tile and
feather-blending the results.

Self-contained: only imports from comfy core (`nodes`, `comfy.sample`,
`comfy.samplers`, `comfy.utils`) — no third-party-pack imports per the
PromptLibrary policy.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.utils
import nodes

try:
    from .runtime import oom_safe_vae_decode, oom_safe_vae_encode
except ImportError:
    from runtime import oom_safe_vae_decode, oom_safe_vae_encode


_log = logging.getLogger("GrimmRibbity.Upscale")

_COMMON_KSAMPLER = None
_CONTROLNET_APPLY_ADVANCED = None
_IMAGE_UPSCALE_WITH_MODEL = None


def _resolve_comfy_helpers() -> None:
    """Resolve `nodes` singletons we use in the hot path. ComfyUI's `nodes`
    module isn't fully populated at our import time, so we resolve once on
    first call."""
    global _COMMON_KSAMPLER, _CONTROLNET_APPLY_ADVANCED, _IMAGE_UPSCALE_WITH_MODEL
    if _COMMON_KSAMPLER is not None:
        return
    _COMMON_KSAMPLER = nodes.common_ksampler
    _CONTROLNET_APPLY_ADVANCED = nodes.ControlNetApplyAdvanced()
    _IMAGE_UPSCALE_WITH_MODEL = nodes.ImageUpscaleWithModel()


@dataclass(frozen=True)
class _Tile:
    """A single tile rectangle in image-pixel space.

    `inner_*` is the region this tile is responsible for in the final
    composite. `outer_*` includes context padding for the sampler — only
    the inner region is blended back, but the sampler sees the padded
    crop so it has neighbour context."""
    inner_x: int
    inner_y: int
    inner_w: int
    inner_h: int
    outer_x: int
    outer_y: int
    outer_w: int
    outer_h: int


def _compute_tile_grid(
    img_w: int, img_h: int, tile_w: int, tile_h: int, padding: int,
    force_uniform: bool,
) -> list[_Tile]:
    """Cover the image with non-overlapping inner tiles plus a ring of
    context padding for sampling. The padding extends outside the inner
    rect on all four sides, clamped to image bounds.

    `force_uniform=True` (USDU's `force_uniform_tiles`) keeps the inner
    tile size constant by sliding the last column/row inward to land
    exactly on the right/bottom edge — avoids a thin trailing strip
    that gets a different sampler signature than the rest. The cost is
    that the second-to-last row/col may overlap the last by a few px;
    feather blending handles that fine.
    """
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile dims must be positive")
    cols = max(1, (img_w + tile_w - 1) // tile_w)
    rows = max(1, (img_h + tile_h - 1) // tile_h)
    tiles: list[_Tile] = []
    for ry in range(rows):
        for cx in range(cols):
            x = cx * tile_w
            y = ry * tile_h
            w = min(tile_w, img_w - x)
            h = min(tile_h, img_h - y)
            if force_uniform and cx == cols - 1 and w < tile_w and img_w >= tile_w:
                x = img_w - tile_w
                w = tile_w
            if force_uniform and ry == rows - 1 and h < tile_h and img_h >= tile_h:
                y = img_h - tile_h
                h = tile_h
            ox = max(0, x - padding)
            oy = max(0, y - padding)
            ow = min(img_w - ox, w + padding + (x - ox))
            oh = min(img_h - oy, h + padding + (y - oy))
            tiles.append(_Tile(x, y, w, h, ox, oy, ow, oh))
    return tiles


def _make_feather_mask(h: int, w: int, blur: int, device, dtype=torch.float32) -> torch.Tensor:
    """Build a HxW mask of 1s with a soft falloff around the edges.

    `blur` is the feather-radius in pixels; the mask transitions from 1
    to 0 over that distance from each edge. Used to crossfade overlapping
    tiles in the composite — the running output is multiplied by (1 -
    feather) on the destination region and the new tile by feather, so
    seam pixels average between the two."""
    if blur <= 0 or w <= 1 or h <= 1:
        return torch.ones((h, w), device=device, dtype=dtype)
    blur = min(blur, w // 2, h // 2)
    if blur <= 0:
        return torch.ones((h, w), device=device, dtype=dtype)
    ramp_x = torch.linspace(0, 1, blur, device=device, dtype=dtype)
    row = torch.ones(w, device=device, dtype=dtype)
    row[:blur] = ramp_x
    row[-blur:] = ramp_x.flip(0)
    col = torch.ones(h, device=device, dtype=dtype)
    col[:blur] = ramp_x
    col[-blur:] = ramp_x.flip(0)
    return col[:, None] * row[None, :]


def _round_to_8(v: int) -> int:
    """SDXL VAE works in multiples of 8 px (latent stride). Round up to
    avoid encode/decode reshape errors on odd tile sizes."""
    return ((v + 7) // 8) * 8


def _scale_image(image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """Lanczos resize an IMAGE tensor [B,H,W,C] to (target_h, target_w).
    Falls back to bicubic if Lanczos isn't available in this comfy version."""
    samples = image.movedim(-1, 1)  # [B,H,W,C] -> [B,C,H,W]
    try:
        scaled = comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", "disabled")
    except Exception:
        scaled = F.interpolate(samples, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return scaled.movedim(1, -1).clamp(0, 1)


def _apply_controlnet(
    positive, negative, control_net, hint_image: torch.Tensor,
    strength: float, start_pct: float, end_pct: float,
):
    """Wrap nodes.ControlNetApplyAdvanced for the conditioning pair."""
    out = _CONTROLNET_APPLY_ADVANCED.apply_controlnet(
        positive=positive, negative=negative, control_net=control_net,
        image=hint_image, strength=strength,
        start_percent=start_pct, end_percent=end_pct,
    )
    return out[0], out[1]


class GrimmRibbityUpscaleSDXL:
    """Single-node tile diffusion upscaler for SDXL.

    Pre-upscales the input (via upscale_model if wired, otherwise Lanczos
    to upscale_by), applies an SDXL tile ControlNet to the conditioning
    if wired, then walks the canvas tile-by-tile running a low-denoise
    sampler. Feather-blends each tile back into the running composite.

    Defaults match the Magnific Precise V2 / Clarity recipe — denoise
    0.25, CFG 4.0, CN strength 0.5, 512 px tiles.
    """

    DESCRIPTION = (
        "Tile diffusion upscaler. Wraps the LoadControlNet + ApplyControlNet "
        "+ UltimateSDUpscale chain into one node with all knobs exposed. "
        "Internal ControlNet apply uses the pre-upscaled image as the hint "
        "(same as wiring Apply ControlNet externally with strength 0.5). "
        "Set bypass=True to skip the diffusion pass and return the upscaled "
        "image directly."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to upscale."}),
                "model": ("MODEL", {"tooltip": "SDXL UNet (post-LoRA, post-anchor — whatever you'd "
                                                "feed your sampler)."}),
                "vae": ("VAE", {}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning. CN apply (if "
                                                          "wired) happens inside this node — pass raw "
                                                          "CLIPTextEncode output here."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),

                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Final scale relative to input. 2.0 = double dimensions."}),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much the sampler is allowed to deviate per tile. 0.20-0.30 is the "
                               "Magnific-style sweet spot — high enough to inject detail, low enough "
                               "to preserve identity. >0.40 starts hallucinating."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Sampler steps per tile. 20 is plenty at low denoise."}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance per tile. 4.0 keeps the prompt subtle "
                               "(detail injection, not generation)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),

                "tile_width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Inner tile width in upscaled-image px. 512 is the SDXL sweet spot."}),
                "tile_height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8,
                    "tooltip": "Context px around each tile that the sampler sees but doesn't "
                               "contribute to the composite. Reduces seams."}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64,
                    "tooltip": "Feather-blend radius between adjacent tiles. Higher = softer seams "
                               "but more blur of detail at boundaries."}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True,
                    "tooltip": "Slide last column/row inward so all tiles are the same size. "
                               "Avoids a thin trailing strip with different sampler behaviour."}),

                "cn_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "ControlNet strength. Ignored when control_net is unwired. 0.5 is the "
                               "Clarity/Magnific recipe; raise toward 0.7 if the upscale drifts."}),
                "cn_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cn_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),

                "seam_fix_mode": (["none", "band_pass"], {"default": "none",
                    "tooltip": "Optional second pass over horizontal+vertical seams between tiles. "
                               "'band_pass' samples thin bands centered on each seam to smooth "
                               "lingering discontinuities. Costs ~25-40% more time."}),
                "seam_fix_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise for the seam-fix pass. Higher than the main pass since "
                               "seam bands need more deviation to fix discontinuities."}),
                "seam_fix_width": ("INT", {"default": 64, "min": 16, "max": 256, "step": 8,
                    "tooltip": "Width (px) of each seam-fix band."}),

                "bypass": ("BOOLEAN", {"default": False,
                    "tooltip": "Skip diffusion: just return the pre-upscaled image. Useful for "
                               "comparing the upscale-model-only output against the full pipeline."}),
            },
            "optional": {
                "control_net": ("CONTROL_NET", {"tooltip": "Optional SDXL tile ControlNet. When "
                                                            "wired, hint = pre-upscaled image."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional pre-upscaler (e.g. "
                                                                "4x-AnimeSharp). Run before tile "
                                                                "diffusion. If unwired, Lanczos."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "GrimmRibbity/Upscale"

    def upscale(
        self, image, model, vae, positive, negative,
        upscale_by, denoise, steps, cfg, sampler_name, scheduler, seed,
        tile_width, tile_height, tile_padding, mask_blur, force_uniform_tiles,
        cn_strength, cn_start_percent, cn_end_percent,
        seam_fix_mode, seam_fix_denoise, seam_fix_width,
        bypass,
        control_net=None, upscale_model=None,
    ):
        _resolve_comfy_helpers()

        device = image.device
        b, src_h, src_w, _ = image.shape
        target_w = _round_to_8(int(src_w * upscale_by))
        target_h = _round_to_8(int(src_h * upscale_by))

        if upscale_model is not None:
            pre = _IMAGE_UPSCALE_WITH_MODEL.upscale(upscale_model, image)[0]
            if pre.shape[1] != target_h or pre.shape[2] != target_w:
                pre = _scale_image(pre, target_w, target_h)
        else:
            pre = _scale_image(image, target_w, target_h)

        if bypass:
            return (pre,)

        if control_net is not None and cn_strength > 0:
            positive, negative = _apply_controlnet(
                positive, negative, control_net, pre,
                cn_strength, cn_start_percent, cn_end_percent,
            )

        tw = _round_to_8(tile_width)
        th = _round_to_8(tile_height)
        tiles = _compute_tile_grid(target_w, target_h, tw, th, tile_padding, force_uniform_tiles)
        _log.info("[Upscale] %dx%d → %dx%d, %d tiles (%dx%d each, pad %d)",
                  src_w, src_h, target_w, target_h, len(tiles), tw, th, tile_padding)

        out = pre.clone()
        pbar = comfy.utils.ProgressBar(len(tiles) + (len(tiles) if seam_fix_mode != "none" else 0))

        try:
            import comfy.model_management as _mm
        except Exception:
            _mm = None

        for i, t in enumerate(tiles):
            # Soft-cancel between tiles — comfy's interrupt checks fire inside
            # common_ksampler too, but encode/composite/decode are not covered.
            # Catching here makes the Cancel button responsive throughout a
            # 30-tile upscale rather than only mid-sample.
            if _mm is not None:
                with contextlib.suppress(AttributeError):
                    _mm.throw_exception_if_processing_interrupted()
            tile_seed = (seed + i) & 0xFFFFFFFFFFFFFFFF
            crop = pre[:, t.outer_y:t.outer_y + t.outer_h, t.outer_x:t.outer_x + t.outer_w, :]
            cw = _round_to_8(crop.shape[2])
            ch = _round_to_8(crop.shape[1])
            if cw != crop.shape[2] or ch != crop.shape[1]:
                crop = _scale_image(crop, cw, ch)

            latent_in = oom_safe_vae_encode(vae, crop[:, :, :, :3], mode="true")
            sampled = _COMMON_KSAMPLER(
                model, tile_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_in, denoise=denoise,
            )[0]
            decoded = oom_safe_vae_decode(vae, sampled, mode="true")
            if decoded.shape[1] != t.outer_h or decoded.shape[2] != t.outer_w:
                decoded = _scale_image(decoded, t.outer_w, t.outer_h)

            inner_dx = t.inner_x - t.outer_x
            inner_dy = t.inner_y - t.outer_y
            inner_patch = decoded[:, inner_dy:inner_dy + t.inner_h, inner_dx:inner_dx + t.inner_w, :]
            mask = _make_feather_mask(t.inner_h, t.inner_w, mask_blur, device, dtype=out.dtype)
            mask = mask[None, :, :, None]
            dst = out[:, t.inner_y:t.inner_y + t.inner_h, t.inner_x:t.inner_x + t.inner_w, :]
            out[:, t.inner_y:t.inner_y + t.inner_h, t.inner_x:t.inner_x + t.inner_w, :] = (
                dst * (1.0 - mask) + inner_patch * mask
            )
            pbar.update(1)

        if seam_fix_mode == "band_pass":
            out = self._seam_band_pass(
                out, model, vae, positive, negative,
                tile_width=tw, tile_height=th,
                steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                seed=seed + len(tiles), denoise=seam_fix_denoise, band_width=seam_fix_width,
                mask_blur=mask_blur, pbar=pbar,
            )

        return (out.clamp(0, 1),)

    def _seam_band_pass(
        self, image, model, vae, positive, negative,
        *, tile_width, tile_height, steps, cfg, sampler_name, scheduler,
        seed, denoise, band_width, mask_blur, pbar,
    ):
        """Run thin sampler bands centered on each interior tile-grid seam.

        Picks up where Linear-mode tile sampling leaves off — feather
        blending fades artifacts at boundaries but doesn't *fix* them
        if neighbour tiles diverged enough that the average looks blurry.
        Band sampling lets diffusion re-knit the seam itself."""
        b, h, w, c = image.shape
        out = image.clone()
        bw = _round_to_8(band_width)
        try:
            import comfy.model_management as _mm
        except Exception:
            _mm = None

        seams_x = list(range(tile_width, w, tile_width))
        for sx in seams_x:
            x0 = max(0, sx - bw // 2)
            x1 = min(w, x0 + bw)
            x0 = max(0, x1 - bw)
            for y0 in range(0, h, tile_height):
                if _mm is not None:
                    with contextlib.suppress(AttributeError):
                        _mm.throw_exception_if_processing_interrupted()
                y1 = min(h, y0 + tile_height)
                yh = _round_to_8(y1 - y0)
                if yh < 64:
                    continue
                self._sample_band_into(out, model, vae, positive, negative,
                                       x0, y0, x1 - x0, yh, steps, cfg,
                                       sampler_name, scheduler, seed, denoise, mask_blur)
                seed += 1
                pbar.update(1)

        seams_y = list(range(tile_height, h, tile_height))
        for sy in seams_y:
            y0 = max(0, sy - bw // 2)
            y1 = min(h, y0 + bw)
            y0 = max(0, y1 - bw)
            for x0 in range(0, w, tile_width):
                if _mm is not None:
                    with contextlib.suppress(AttributeError):
                        _mm.throw_exception_if_processing_interrupted()
                x1 = min(w, x0 + tile_width)
                xw = _round_to_8(x1 - x0)
                if xw < 64:
                    continue
                self._sample_band_into(out, model, vae, positive, negative,
                                       x0, y0, xw, y1 - y0, steps, cfg,
                                       sampler_name, scheduler, seed, denoise, mask_blur)
                seed += 1
                pbar.update(1)

        return out

    def _sample_band_into(
        self, image, model, vae, positive, negative,
        x, y, w, h, steps, cfg, sampler_name, scheduler, seed, denoise, mask_blur,
    ):
        """In-place: sample one band rectangle and feather-blend it back."""
        crop = image[:, y:y + h, x:x + w, :]
        latent_in = oom_safe_vae_encode(vae, crop[:, :, :, :3], mode="true")
        sampled = _COMMON_KSAMPLER(
            model, seed & 0xFFFFFFFFFFFFFFFF, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_in, denoise=denoise,
        )[0]
        decoded = oom_safe_vae_decode(vae, sampled, mode="true")
        if decoded.shape[1] != h or decoded.shape[2] != w:
            decoded = _scale_image(decoded, w, h)
        mask = _make_feather_mask(h, w, mask_blur, image.device, dtype=image.dtype)
        mask = mask[None, :, :, None]
        dst = image[:, y:y + h, x:x + w, :]
        image[:, y:y + h, x:x + w, :] = dst * (1.0 - mask) + decoded * mask


NODE_CLASS_MAPPINGS = {"GrimmRibbityUpscaleSDXL": GrimmRibbityUpscaleSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"GrimmRibbityUpscaleSDXL": "GrimmRibbity — Upscale SDXL"}
