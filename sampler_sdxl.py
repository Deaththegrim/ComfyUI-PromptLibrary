"""GrimmRibbity SDXL Sampler + HiResFix script.

A pair of nodes that replaces the efficiency-nodes `KSampler SDXL (Eff.)`
+ `HighRes-Fix Script` combo. Lessons learned:

- Output values must be wire-compatible with the rest of ComfyUI. The
  efficiency node emits sentinel values (-1 seed, weird shape state) that
  trip wire validation downstream. Our sampler emits ordinary values.
- The HiResFix is a separate script node that the sampler runs after its
  primary pass. Same shape as efficiency-nodes, so the workflow pattern
  feels familiar, but with a clean re-implementation.
- Everything bundled into the sampler is just thin wrappers around
  ComfyUI's stable internals (load_checkpoint_guess_config, the same
  KSampler core that drives the stock node, VAEDecode, etc.) — we are
  not reinventing the diffusion math.

The two nodes share a `GRIMM_SDXL_SCRIPT` pipe type. The sampler's
optional `script` input takes one of these and runs the script's
`apply()` after primary sampling.
"""

from __future__ import annotations

from typing import Any

import torch

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import nodes  # for common_ksampler


# Custom pipe type. ComfyUI uses string types for wire compatibility — any
# unique string works as long as the same string is used on both sides.
GRIMM_SDXL_SCRIPT_TYPE = "GRIMM_SDXL_SCRIPT"


def _encode_sdxl(clip, text_g: str, text_l: str,
                 width: int, height: int,
                 target_width: int, target_height: int,
                 crop_w: int = 0, crop_h: int = 0):
    """Replicates comfy_extras CLIPTextEncodeSDXL.execute without the v3 schema
    layer, so we can call it from a vanilla v2 node class."""
    tokens = clip.tokenize(text_g)
    tokens["l"] = clip.tokenize(text_l)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    return clip.encode_from_tokens_scheduled(tokens, add_dict={
        "width": width, "height": height,
        "crop_w": crop_w, "crop_h": crop_h,
        "target_width": target_width, "target_height": target_height,
    })


def _empty_latent(width: int, height: int, batch_size: int):
    import comfy.model_management
    latent = torch.zeros(
        [batch_size, 4, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def _vae_decode(vae, latent):
    return vae.decode(latent["samples"])


def _vae_encode(vae, image):
    return {"samples": vae.encode(image)}


def _latent_upscale_by(latent: dict, scale: float, method: str = "bislerp") -> dict:
    """Pure latent-space upscale — fast, no model needed, lower quality than
    a model upscaler but plenty for low-strength HiResFix passes."""
    samples = latent["samples"]
    width = round(samples.shape[-1] * scale)
    height = round(samples.shape[-2] * scale)
    out = latent.copy()
    out["samples"] = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return out


_LATENT_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


# ---------------------------------------------------------------------------
# HiRes-Fix script
# ---------------------------------------------------------------------------


class GrimmRibbityHiResFixScript:
    """Plug into the GrimmRibbity SDXL Sampler's `script` input. After the
    primary sample completes, the sampler runs this script's apply() with
    the primary pass's MODEL, CLIP, VAE, conditioning, and latent. Returns
    a refined latent + image."""

    DESCRIPTION = (
        "Two-pass HiRes-Fix script for the GrimmRibbity SDXL Sampler. After "
        "the primary sample finishes, this script upscales the latent and "
        "runs a second sampling pass at lower denoise. Wire its output into "
        "the sampler's `script` input. Use latent-space upscale (fast, "
        "default) or set upscale_method to 'model' and provide an upscale "
        "model for higher quality."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Upscale multiplier. 1.5 = 1.5× larger in both dimensions."}),
                "upscale_method": (_LATENT_UPSCALE_METHODS + ["model"], {
                    "tooltip": "Latent-space methods are fast (bislerp is the smoothest). "
                               "'model' uses a wired upscale_model on the decoded image, "
                               "encodes back to latent — higher quality, slower."}),
                "hires_steps": ("INT", {"default": 12, "min": 1, "max": 200,
                    "tooltip": "Sampling steps for the second pass."}),
                "hires_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much detail the second pass adds. 0.5 is a good middle "
                               "ground; below 0.3 the upscale dominates, above 0.6 you risk "
                               "the AI inventing new content."}),
                "hires_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG for the second pass. Often slightly lower than the "
                               "primary pass to avoid burn-in."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Required if upscale_method='model'. "
                                                                "Wire from a Load Upscale Model node."}),
                "positive_g_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override the positive G prompt for the second pass. Useful for "
                               "adding 'highly detailed, sharp focus' style refinements only at "
                               "the upscale stage. Empty = reuse primary."}),
                "positive_l_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override positive L prompt for the second pass. Empty = reuse primary."}),
                "negative_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override negative prompt for the second pass. Empty = reuse primary."}),
            },
        }

    RETURN_TYPES = (GRIMM_SDXL_SCRIPT_TYPE,)
    RETURN_NAMES = ("script",)
    OUTPUT_TOOLTIPS = (
        "Pipe to wire into the GrimmRibbity SDXL Sampler's `script` input.",
    )
    FUNCTION = "build"
    CATEGORY = "utils"

    def build(self, upscale_by, upscale_method, hires_steps, hires_denoise, hires_cfg,
              upscale_model=None, positive_g_override="", positive_l_override="",
              negative_override=""):
        if upscale_method == "model" and upscale_model is None:
            # Fail loudly — silent fallback would be a worse surprise than
            # an explicit error, since the user explicitly chose the model path.
            raise ValueError("HiResFixScript: upscale_method='model' requires an "
                             "upscale_model input. Wire one in or pick a latent method.")
        return ({
            "kind": "hires_fix",
            "upscale_by": float(upscale_by),
            "upscale_method": upscale_method,
            "hires_steps": int(hires_steps),
            "hires_denoise": float(hires_denoise),
            "hires_cfg": float(hires_cfg),
            "upscale_model": upscale_model,
            "positive_g_override": positive_g_override,
            "positive_l_override": positive_l_override,
            "negative_override": negative_override,
        },)


def _apply_hires_fix(script: dict, *, model, clip, vae, positive, negative, latent, seed,
                     primary_sampler_name: str, primary_scheduler: str,
                     width: int, height: int,
                     primary_text_g: str, primary_text_l: str, primary_negative: str):
    """Run the HiResFix second pass and return a new (latent, image) tuple.
    Called from the SDXL Sampler after its primary pass."""
    upscale_by = script["upscale_by"]
    method = script["upscale_method"]

    if method == "model":
        # Decode → upscale image with model → encode back to latent.
        # Adds two VAE round-trips but the upscale model usually wins.
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel  # noqa
        # Use comfy core's tiled upscale path via a manual implementation — the
        # ImageUpscaleWithModel class is v3-schema-style and awkward to call
        # outside the runtime, so do the same operation by hand.
        import comfy.model_management
        upscale_model = script["upscale_model"]
        device = comfy.model_management.get_torch_device()
        memory_required = comfy.model_management.module_size(upscale_model.model)
        decoded = _vae_decode(vae, latent)
        memory_required += (512 * 512 * 3) * decoded.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += decoded.nelement() * decoded.element_size()
        comfy.model_management.free_memory(memory_required, device)
        upscale_model.to(device)
        in_img = decoded.movedim(-1, -3).to(device)
        tile, overlap = 512, 32
        steps_total = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
            in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
        pbar = comfy.utils.ProgressBar(steps_total)
        upscaled = comfy.utils.tiled_scale(
            in_img, lambda a: upscale_model(a.float()),
            tile_x=tile, tile_y=tile, overlap=overlap,
            upscale_amount=upscale_model.scale,
            pbar=pbar,
            output_device=comfy.model_management.intermediate_device(),
        )
        upscale_model.to(comfy.model_management.vae_offload_device())
        upscaled = upscaled.movedim(-3, -1)
        # Crop down to upscale_by * orig_size (the model has its own native
        # scale, e.g. 4×, so we resample down to the user's chosen ratio).
        target_w = round(decoded.shape[-2] * upscale_by)
        target_h = round(decoded.shape[-3] * upscale_by)
        if upscaled.shape[-2] != target_w or upscaled.shape[-3] != target_h:
            up = upscaled.movedim(-1, -3)
            up = comfy.utils.common_upscale(up, target_w, target_h, "bislerp", "disabled")
            upscaled = up.movedim(-3, -1)
        upscaled_latent = _vae_encode(vae, upscaled.clamp(0, 1))
    else:
        upscaled_latent = _latent_upscale_by(latent, upscale_by, method)

    # Build conditioning for the second pass — reuse primary unless overridden.
    new_g = script["positive_g_override"].strip() or primary_text_g
    new_l = script["positive_l_override"].strip() or primary_text_l
    new_neg = script["negative_override"].strip() or primary_negative
    samples = upscaled_latent["samples"]
    new_w = samples.shape[-1] * 8
    new_h = samples.shape[-2] * 8
    if (new_g == primary_text_g and new_l == primary_text_l):
        # Reuse primary conditioning verbatim — saves a re-encode.
        positive_2 = positive
    else:
        positive_2 = _encode_sdxl(clip, new_g, new_l,
                                   width=new_w, height=new_h,
                                   target_width=new_w, target_height=new_h)
    if new_neg == primary_negative:
        negative_2 = negative
    else:
        negative_2 = _encode_sdxl(clip, new_neg, new_neg,
                                   width=new_w, height=new_h,
                                   target_width=new_w, target_height=new_h)

    final_latent_tuple = nodes.common_ksampler(
        model, seed, script["hires_steps"], script["hires_cfg"],
        primary_sampler_name, primary_scheduler,
        positive_2, negative_2, upscaled_latent,
        denoise=script["hires_denoise"],
    )
    final_latent = final_latent_tuple[0]
    final_image = _vae_decode(vae, final_latent)
    return final_latent, final_image


# ---------------------------------------------------------------------------
# SDXL Sampler
# ---------------------------------------------------------------------------


class GrimmRibbitySamplerSDXL:
    """Monolithic SDXL sampler: checkpoint loader + dual CLIP encode + KSampler
    + VAE decode + optional HiResFix script — all in one node. Outputs the
    final image, the final latent, plus the loaded MODEL/CLIP/VAE so a
    downstream node can use them without reloading."""

    DESCRIPTION = (
        "Replacement for KSampler SDXL (Eff.). Loads the checkpoint, encodes "
        "dual SDXL prompts (text_g + text_l), samples, and decodes the image "
        "in one node. Wire a HiResFix Script into the optional `script` input "
        "to run a second upscale pass. Outputs are wire-compatible with the "
        "rest of ComfyUI — no -1 sentinels, no broken seeds."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "SDXL checkpoint to load (one of your installed checkpoints)."}),
                "positive_g": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "SDXL positive G prompt — typically the longer descriptive text. "
                               "Goes through the OpenCLIP-G text encoder."}),
                "positive_l": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "SDXL positive L prompt — typically tags / shorter cues. "
                               "Goes through the CLIP-L encoder. Often duplicates positive_g."}),
                "negative": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Negative prompt. Used for both G and L encoders."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64,
                    "tooltip": "Output width in pixels (before HiResFix)."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64,
                    "tooltip": "Output height in pixels (before HiResFix)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True,
                    "tooltip": "Random seed. Same seed + same prompts = reproducible output."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                    "tooltip": "Primary-pass sampling steps."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance scale for the primary pass."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampling algorithm. dpmpp_2m / euler are common SDXL choices."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Noise schedule. karras pairs well with dpmpp samplers."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Primary-pass denoise. 1.0 = full txt2img. Lower = preserve a "
                               "wired latent_image (img2img-style)."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                    "tooltip": "How many images to generate per queue."}),
            },
            "optional": {
                "latent_image": ("LATENT", {"tooltip": "Optional latent for img2img. Empty "
                                                        "skipped — uses an empty latent of "
                                                        "the configured width × height."}),
                "script": (GRIMM_SDXL_SCRIPT_TYPE, {"tooltip": "Optional GrimmRibbity script "
                                                                "pipe (e.g. HiResFix). Runs "
                                                                "after the primary sample."}),
                "vae_override": ("VAE", {"tooltip": "Optional external VAE. Overrides the VAE "
                                                     "loaded from the checkpoint."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MODEL", "CLIP", "VAE", "INT")
    RETURN_NAMES = ("image", "latent", "model", "clip", "vae", "seed")
    OUTPUT_TOOLTIPS = (
        "Final decoded image (after HiResFix if a script was wired).",
        "Final latent (after HiResFix if a script was wired).",
        "The loaded MODEL — wire into another sampler if you want a chained pass.",
        "The loaded CLIP — wire into a text-encode node for another pass.",
        "The loaded VAE — wire into VAEEncode/Decode if needed elsewhere.",
        "The seed actually used. Wire into a logger or CivitaiSaveImage.seed_override.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, ckpt_name, positive_g, positive_l, negative,
               width, height, seed, steps, cfg, sampler_name, scheduler,
               denoise, batch_size,
               latent_image=None, script=None, vae_override=None):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        loaded = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = loaded[:3]
        if vae_override is not None:
            vae = vae_override

        positive_cond = _encode_sdxl(clip, positive_g, positive_l,
                                      width=width, height=height,
                                      target_width=width, target_height=height)
        negative_cond = _encode_sdxl(clip, negative, negative,
                                      width=width, height=height,
                                      target_width=width, target_height=height)

        if latent_image is None:
            latent_image = _empty_latent(width, height, batch_size)

        primary_latent_tuple = nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive_cond, negative_cond, latent_image, denoise=denoise,
        )
        latent_out = primary_latent_tuple[0]
        image_out = _vae_decode(vae, latent_out)

        if script is not None:
            kind = script.get("kind") if isinstance(script, dict) else None
            if kind == "hires_fix":
                latent_out, image_out = _apply_hires_fix(
                    script, model=model, clip=clip, vae=vae,
                    positive=positive_cond, negative=negative_cond,
                    latent=latent_out, seed=seed,
                    primary_sampler_name=sampler_name,
                    primary_scheduler=scheduler,
                    width=width, height=height,
                    primary_text_g=positive_g, primary_text_l=positive_l,
                    primary_negative=negative,
                )
            else:
                print(f"[GrimmRibbitySamplerSDXL] unknown script kind {kind!r}; skipped")

        return (image_out, latent_out, model, clip, vae, int(seed))
