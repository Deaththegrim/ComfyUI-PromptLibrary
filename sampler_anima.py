"""GrimmRibbity Anima Sampler — for Qwen-Image and other flow-matching
anime models. Takes pre-encoded CONDITIONING + MODEL + LATENT directly
(no tuple, no in-node text encoding) so it works regardless of the text
encoder architecture (Qwen3, T5, single-CLIP, dual-CLIP, …).

Use this instead of GrimmRibbity SDXL Sampler when:

  - Your checkpoint is a Qwen-Image variant (16-channel 5D latents)
  - Your checkpoint is Flux / SD3 / any flow-matching base
  - Your text encoder isn't SDXL's CLIP-G + CLIP-L pair

The accompanying GrimmRibbityAnimaHiResFixScript is the safe HiResFix
plug-in: it uses only Comfy's tensor-shape-agnostic interpolation
methods (nearest-exact / bilinear / bicubic / bislerp / area), never
the SDXL-specific neural latent upscalers (city96.* / ttl_nn.SDXL
expect 4-channel 4D SDXL latents and crash on Qwen's 16-channel 5D).

We're not reimplementing diffusion math — every numerical operation
delegates to ComfyUI core (nodes.common_ksampler, comfy.utils.common_upscale,
the VAE's decode / decode_tiled).
"""

from __future__ import annotations

import torch

import comfy.samplers
import comfy.utils
import nodes


GRIMM_ANIMA_SCRIPT_TYPE = "GRIMM_ANIMA_SCRIPT"

_VAE_DECODE_MODES = ["true", "true (tiled)", "false"]
_INTERPOLATION_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


_AUTO_TILE_LATENT_THRESHOLD = 192  # latent pixels — see sampler_sdxl._smart_vae_decode


def _vae_decode(vae, latent, *, mode: str = "true"):
    """Decode honouring the user's vae_decode mode. 'false' returns None.
    'true' auto-promotes to tiled when the latent's longest dim exceeds
    _AUTO_TILE_LATENT_THRESHOLD — saves the user from picking 'true (tiled)'
    manually for HiResFix outputs that would OOM the non-tiled path."""
    if mode == "false" or vae is None:
        return None
    samples = latent["samples"]
    latent_max = max(samples.shape[-1], samples.shape[-2])
    if mode == "true" and latent_max > _AUTO_TILE_LATENT_THRESHOLD and hasattr(vae, "decode_tiled"):
        return vae.decode_tiled(samples, tile_x=512, tile_y=512, overlap=64)
    if mode == "true (tiled)" and hasattr(vae, "decode_tiled"):
        return vae.decode_tiled(samples, tile_x=512, tile_y=512, overlap=64)
    return vae.decode(samples)


def _interpolation_upscale(latent: dict, scale: float, method: str) -> dict:
    """Shape-agnostic latent upscale. Works with 4D SDXL latents AND with
    Qwen-Image's 5D `[B, C, T, H, W]` layout — F.interpolate (which
    common_upscale wraps) handles both. Avoids any model-specific neural
    upscaler that would expect a particular channel count or rank."""
    samples = latent["samples"]
    width = round(samples.shape[-1] * scale)
    height = round(samples.shape[-2] * scale)
    out = latent.copy()
    out["samples"] = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return out


# ---------------------------------------------------------------------------
# HiResFix script
# ---------------------------------------------------------------------------


class GrimmRibbityAnimaHiResFixScript:
    """Safe HiResFix companion for the Anima Sampler. Uses Comfy's
    tensor-shape-agnostic interpolation methods only — no SDXL-specific
    neural latent upscalers, so it works with any latent shape (4D SDXL
    or 5D Qwen / Flux / video). Reuses the upstream CONDITIONING
    verbatim for the second pass; doesn't try to re-encode the prompt
    (which would require knowing the encoder's API)."""

    DESCRIPTION = (
        "HiRes-Fix script for the GrimmRibbity Anima Sampler. After the "
        "primary sample, upscales the latent via shape-agnostic Comfy "
        "interpolation (nearest-exact / bilinear / area / bicubic / "
        "bislerp) and runs a second sampling pass at lower denoise. "
        "Reuses the upstream conditioning verbatim — no re-encoding, so "
        "the encoder's architecture (Qwen3 / T5 / CLIP) doesn't matter. "
        "1-5 iterations. No neural upscalers — those are SDXL-specific."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_method": (_INTERPOLATION_METHODS, {
                    "tooltip": "Latent-space upscale method. bislerp is the smoothest default. "
                               "All five work on any latent shape (4D SDXL or 5D Qwen)."}),
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Final cumulative upscale multiplier across all iterations."}),
                "use_same_seed": ("BOOLEAN", {"default": True,
                    "tooltip": "Use the primary sampler's seed for the hires pass too. "
                               "Off = use the explicit `seed` field below."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True,
                    "tooltip": "Hires-pass seed. Ignored if use_same_seed=True. "
                               "Iteration N adds N for stable variation across passes."}),
                "hires_steps": ("INT", {"default": 12, "min": 1, "max": 200,
                    "tooltip": "Sampling steps per hires iteration."}),
                "hires_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much detail each hires pass adds. 0.5 is a safe middle. "
                               "<0.3 the upscale dominates; >0.6 the model invents new content."}),
                "hires_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG for hires passes. Often slightly lower than primary."}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 5,
                    "tooltip": "Number of upscale + sample iterations. Each iteration scales "
                               "by upscale_by^(1/iterations) so the cumulative scale equals "
                               "upscale_by."}),
            },
        }

    RETURN_TYPES = (GRIMM_ANIMA_SCRIPT_TYPE,)
    RETURN_NAMES = ("script",)
    OUTPUT_TOOLTIPS = ("Pipe to wire into the GrimmRibbity Anima Sampler's `script` input.",)
    FUNCTION = "build"
    CATEGORY = "GrimmRibbity/Anima"

    def build(self, upscale_method, upscale_by, use_same_seed, seed, hires_steps,
              hires_denoise, hires_cfg, iterations):
        return ({
            "kind": "anima_hires_fix",
            "upscale_method": upscale_method,
            "upscale_by": float(upscale_by),
            "use_same_seed": bool(use_same_seed),
            "seed": int(seed),
            "hires_steps": int(hires_steps),
            "hires_denoise": float(hires_denoise),
            "hires_cfg": float(hires_cfg),
            "iterations": max(1, int(iterations)),
        },)


def _apply_anima_hires_fix(script: dict, *,
                            model, positive, negative, latent,
                            primary_seed: int, primary_sampler_name: str,
                            primary_scheduler: str):
    """Run the Anima HiResFix passes. Returns the final latent. Uses the
    shared _HiresIterPlan for per-iter scale, no-op-skip flag, and the
    progress bar — same bookkeeping as the SDXL HiResFix so behaviour
    stays consistent."""
    # Cross-module borrow for the shared plan + pre-flight warning. SDXL is
    # always present (no torch fallback path on this side either, since
    # this whole function only runs at sample time when torch is loaded).
    from .sampler_sdxl import _HiresIterPlan, _preflight_size_warning

    base_seed = primary_seed if script["use_same_seed"] else script["seed"]

    _preflight_size_warning(latent, script["upscale_by"], sampler="GrimmRibbityAnimaSampler")

    plan = _HiresIterPlan.build(
        total_scale=script["upscale_by"], iterations=script["iterations"])

    cur = latent
    for i in range(plan.iterations):
        if not plan.skip_upscale:
            cur = _interpolation_upscale(cur, plan.per_iter, script["upscale_method"])
        sampled = nodes.common_ksampler(
            model, base_seed + i,
            script["hires_steps"], script["hires_cfg"],
            primary_sampler_name, primary_scheduler,
            positive, negative, cur,
            denoise=script["hires_denoise"],
        )
        cur = sampled[0]
        plan.pbar.update(1)
    return cur


# ---------------------------------------------------------------------------
# Anima Sampler
# ---------------------------------------------------------------------------


class GrimmRibbityAnimaSampler:
    """Sampler for Qwen-Image and other flow-matching anime models. Takes
    MODEL + pre-encoded CONDITIONING + LATENT directly — no SDXL-only
    tuple, no in-node text encoding, no architecture assumptions. Works
    with any model where ComfyUI's stock common_ksampler works.

    Wire it the same way you'd wire stock KSampler:
      UNETLoader / CheckpointLoaderSimple → MODEL
      CLIPLoader → CLIP → CLIPTextEncode → positive / negative CONDITIONING
      EmptyLatentImage / EmptyLatentImageFromPresetsSDXL → LATENT
      VAELoader → optional_vae (for the decode)

    Outputs IMAGE / LATENT / MODEL / seed for chaining. Optional `script`
    input for HiResFix — wire a GrimmRibbity Anima HiResFix Script.
    """

    DESCRIPTION = (
        "Sampler for Qwen-Image / Flux / SD3 / any flow-matching anime "
        "model. Unlike the SDXL Sampler, this one takes MODEL + "
        "pre-encoded CONDITIONING directly — no tuple, no in-node "
        "encoding, no SDXL-specific assumptions. Pair with the Anima "
        "HiResFix Script for safe upscaling that doesn't require "
        "SDXL-specific neural upscalers."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Required. From UNETLoader, CheckpointLoaderSimple, "
                               "or after a Power Lora Loader chain."}),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning. Encode with CLIPTextEncode (or any "
                               "encoder that emits CONDITIONING)."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning."}),
                "latent_image": ("LATENT", {
                    "tooltip": "Empty or img2img latent. Width/height/batch_size are "
                               "determined by this latent."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                        "control_after_generate": True,
                    "tooltip": "Random seed for the primary noise."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                    "tooltip": "Total denoising steps for the primary pass."}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance scale. Flow-matching models often "
                               "want lower CFG (3-5) than SDXL (7-9)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampling algorithm. euler / heun work well for "
                               "flow-matching models."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Noise schedule. simple / sgm_uniform pair well with "
                               "flow-matching."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Primary denoise. 1.0 = full txt2img. Lower = preserve a "
                               "wired latent (img2img-style)."}),
                "vae_decode": (_VAE_DECODE_MODES, {
                    "tooltip": "true: decode every output. true (tiled): tiled decode for "
                               "large images. false: skip decode (returns placeholder image)."}),
            },
            "optional": {
                "optional_vae": ("VAE", {
                    "tooltip": "VAE for the decode. Required unless vae_decode='false'."}),
                "script": (GRIMM_ANIMA_SCRIPT_TYPE, {
                    "tooltip": "Optional Anima script pipe (e.g. Anima HiResFix). Runs "
                               "after the primary pass."}),
                "save_prompt_log": ("BOOLEAN", {"default": False,
                    "tooltip": "When True, append one JSONL line per call to prompt_log_path "
                               "(positive, negative, loras, seed, sampler params, model). "
                               "Default off — flip on for overnight batches you want to grep later."}),
                "prompt_log_path": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "JSONL log file. Empty = <output>/prompt_logs/prompts.jsonl. "
                               "Relative paths root at the ComfyUI output dir; absolute paths "
                               "honoured verbatim."}),
            },
            "hidden": {
                "prompt_trace": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MODEL", "INT")
    RETURN_NAMES = ("image", "latent", "model", "seed")
    OUTPUT_TOOLTIPS = (
        "Final decoded image. 1×1×3 zeros if vae_decode='false'.",
        "Final latent (after HiResFix if a script was wired).",
        "The MODEL passed in — chain into another sampler if needed.",
        "The seed actually used (always non-negative).",
    )
    FUNCTION = "sample"
    CATEGORY = "GrimmRibbity/Anima"
    OUTPUT_NODE = True

    def sample(self, model, positive, negative, latent_image, noise_seed, steps,
               cfg, sampler_name, scheduler, denoise, vae_decode,
               optional_vae=None, script=None,
               save_prompt_log=False, prompt_log_path="", prompt_trace=None):
        if vae_decode != "false" and optional_vae is None:
            raise ValueError(
                "Anima Sampler: vae_decode is enabled but optional_vae is not wired. "
                "Wire a VAE into the optional_vae input, or set vae_decode='false' to "
                "skip decoding (the IMAGE port will then emit a 1×1×3 placeholder)."
            )
        primary = nodes.common_ksampler(
            model, noise_seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=denoise,
        )
        latent_out = primary[0]

        if script is not None:
            kind = script.get("kind") if isinstance(script, dict) else None
            if kind == "anima_hires_fix":
                latent_out = _apply_anima_hires_fix(
                    script, model=model, positive=positive, negative=negative,
                    latent=latent_out, primary_seed=noise_seed,
                    primary_sampler_name=sampler_name,
                    primary_scheduler=scheduler,
                )
            else:
                print(f"[GrimmRibbityAnimaSampler] unknown script kind {kind!r}; skipped")

        image_out = _vae_decode(optional_vae, latent_out, mode=vae_decode)
        if image_out is None:
            image_out = torch.zeros((1, 1, 1, 3))

        if save_prompt_log:
            try:
                from .prompt_log import append_prompt_log, build_record, resolve_log_path
                batch_size = (latent_image.get("samples").shape[0]
                               if isinstance(latent_image, dict)
                               and hasattr(latent_image.get("samples"), "shape") else 1)
                record = build_record(
                    sampler_node="GrimmRibbityAnimaSampler",
                    prompt_trace=prompt_trace,
                    runtime={"seed": int(noise_seed), "steps": steps, "cfg": cfg,
                              "sampler_name": sampler_name, "scheduler": scheduler,
                              "denoise": denoise, "batch_size": batch_size},
                )
                append_prompt_log(record, resolve_log_path(prompt_log_path))
            except Exception as e:
                print(f"[GrimmRibbityAnimaSampler] prompt_log append failed: {e}")

        return (image_out, latent_out, model, int(noise_seed))
