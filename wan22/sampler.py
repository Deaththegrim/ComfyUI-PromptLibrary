"""Wan2.2 two-stage KSampler.

Collapses the standard Wan2.2-A14B i2v workflow — two chained KSamplerAdvanced
nodes, one per expert, joined at a step boundary — into a single node. Same
semantics as wiring:

    KSamplerAdvanced(model_high, add_noise=enable,  start=0,        end=boundary, return_with_leftover=enable)
        -> KSamplerAdvanced(model_low,  add_noise=disable, start=boundary, end=steps,    return_with_leftover=disable)

…just with one less LATENT wire and a sensible default boundary (steps // 2).
"""

from __future__ import annotations

import comfy.samplers
import nodes  # for common_ksampler


_INT_MAX = 0xFFFFFFFFFFFFFFFF


class GrimmRibbityWan22TwoStageKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {
                    "tooltip": "HighNoise expert — runs steps 0 → boundary."}),
                "model_low": ("MODEL", {
                    "tooltip": "LowNoise expert — runs steps boundary → steps."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT", {
                    "tooltip": "From WanImageToVideo / Wan22ImageToVideoLatent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX,
                    "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 2, "max": 1000,
                    "tooltip": "Total denoising steps across both experts."}),
                "boundary_step": ("INT", {"default": 10, "min": 1, "max": 999,
                    "tooltip": "Step at which to hand off HighNoise → LowNoise. "
                               "Wan2.2 community default ≈ steps/2. Clamped to "
                               "[1, steps-1]."}),
                "cfg_high": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0,
                    "step": 0.1, "round": 0.01,
                    "tooltip": "CFG during the HighNoise stage. Wan2.2 i2v "
                               "typical range 3.0–4.0."}),
                "cfg_low": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0,
                    "step": 0.1, "round": 0.01,
                    "tooltip": "CFG during the LowNoise stage. Often equal to "
                               "cfg_high; lower it slightly for less prompt "
                               "adherence in late refinement."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampler applied to both stages. uni_pc and "
                               "euler are the common picks for Wan2.2."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "simple",
                    "tooltip": "Noise schedule. 'simple' or 'beta' are the "
                               "community defaults for Wan2.2 i2v."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Overall denoise strength. 1.0 = full i2v."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("Final denoised video latent — decode with the Wan2.2 VAE.",)
    FUNCTION = "sample"
    CATEGORY = "GrimmRibbity/Wan2.2"
    DESCRIPTION = (
        "Two-stage Wan2.2-A14B sampler. Stage 1 runs the HighNoise expert "
        "from step 0 to boundary_step with full added noise and "
        "return-with-leftover-noise enabled. Stage 2 picks up with the "
        "LowNoise expert from boundary_step to steps, disabling re-noising "
        "and forcing full denoise. Equivalent to wiring two KSamplerAdvanced "
        "nodes by hand."
    )

    def sample(self, model_high, model_low, positive, negative, latent_image,
               seed, steps, boundary_step, cfg_high, cfg_low,
               sampler_name, scheduler, denoise):
        boundary = max(1, min(boundary_step, steps - 1))

        # Stage 1: HighNoise expert, add fresh noise, return with leftover.
        # force_full_denoise=False keeps the partial trajectory so stage 2
        # can pick up from the same latent.
        stage1 = nodes.common_ksampler(
            model_high, seed, steps, cfg_high, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise,
            disable_noise=False,
            start_step=0,
            last_step=boundary,
            force_full_denoise=False,
        )[0]

        # Stage 2: LowNoise expert continues from the partially-denoised
        # latent. No new noise (disable_noise=True), and force_full_denoise
        # so the final step actually clears the residual sigma.
        stage2 = nodes.common_ksampler(
            model_low, seed, steps, cfg_low, sampler_name, scheduler,
            positive, negative, stage1,
            denoise=denoise,
            disable_noise=True,
            start_step=boundary,
            last_step=steps,
            force_full_denoise=True,
        )[0]

        return (stage2,)
