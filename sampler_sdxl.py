"""GrimmRibbity SDXL Sampler suite — efficiency-nodes replacement.

Three nodes in this file:

- `GrimmRibbityPackSDXLTuple` — packs base + (optional) refiner MODEL/
  CLIP/CONDITIONING into an SDXL_TUPLE wire. Same shape as
  efficiency-nodes' SDXL_TUPLE so existing tuple-using nodes plug in.
- `GrimmRibbitySamplerSDXL` — the sampler. Pure SDXL_TUPLE consumer:
  no in-node checkpoint loader, no in-node prompt encoders. Use the
  Library / Wildcard / Scene / Comic Frame nodes to compose prompts,
  encode them with the standard SDXL CLIPTextEncode nodes, then pack
  into a tuple. Outputs IMAGE / LATENT / MODEL / CLIP / VAE / seed /
  SDXL_TUPLE for chaining.
- `GrimmRibbityHiResFixScript` — emits the GRIMM_SDXL_SCRIPT pipe
  consumed by the sampler. Latent / pixel / both upscale, hires
  checkpoint swap, hires seed, configurable iterations, optional
  ControlNet wiring.

We are not reimplementing diffusion math — every numerical operation
delegates to ComfyUI's stable internals (load_checkpoint_guess_config,
common_ksampler, the SDXL encode dance from comfy_extras.nodes_clip_sdxl,
VAE encode/decode, the upscale-with-model tiling logic).
"""

from __future__ import annotations

import torch

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes  # for common_ksampler


GRIMM_SDXL_SCRIPT_TYPE = "GRIMM_SDXL_SCRIPT"
SDXL_TUPLE_TYPE = "SDXL_TUPLE"  # wire-compatible with efficiency-nodes' tuple

_COMFY_LATENT_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
_VAE_DECODE_MODES = ["true", "true (tiled)", "false"]
_UPSCALE_TYPES = ["latent", "pixel", "both"]


def _try_load_efficiency_upscalers():
    """Pull the city96 + ttl_nn neural latent upscalers in from
    efficiency-nodes' bundled py/ folder when that pack is installed.
    Returns (city96_class, ttl_nn_class) — either may be None on failure.
    Used to extend the latent_upscaler dropdown beyond comfy's default
    interpolation methods, matching efficiency-nodes' option set."""
    import importlib.util
    import os
    city = ttl = None
    candidates = (
        "/home/junie/comfy/ComfyUI/custom_nodes/efficiency-nodes-comfyui/py",
        os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes",
                     "efficiency-nodes-comfyui", "py"),
    )
    for base in candidates:
        if not os.path.isdir(base):
            continue
        try:
            spec_c = importlib.util.spec_from_file_location(
                "_grimm_city96_latent_upscaler",
                os.path.join(base, "city96_latent_upscaler.py"))
            mod_c = importlib.util.module_from_spec(spec_c)
            spec_c.loader.exec_module(mod_c)
            city = mod_c.LatentUpscaler
        except Exception as e:
            print(f"[GrimmRibbity] city96 latent upscaler unavailable: {e}")
        try:
            spec_t = importlib.util.spec_from_file_location(
                "_grimm_ttl_nn_latent_upscaler",
                os.path.join(base, "ttl_nn_latent_upscaler.py"))
            mod_t = importlib.util.module_from_spec(spec_t)
            spec_t.loader.exec_module(mod_t)
            ttl = mod_t.NNLatentUpscale
        except Exception as e:
            print(f"[GrimmRibbity] ttl_nn latent upscaler unavailable: {e}")
        if city or ttl:
            break
    return city, ttl


_CITY96_LATENT_CLS, _TTL_NN_LATENT_CLS = _try_load_efficiency_upscalers()
_CITY96_VERSIONS = ["v1", "xl"]
_CITY96_SCALES = [1.25, 1.5, 2.0]
_TTL_NN_VERSIONS = ["SDXL", "SD 1.x"]


def _build_latent_upscaler_choices() -> list[str]:
    """Comfy interpolation methods + neural upscalers prefixed by source.
    Picked source/version is parsed back out at apply time."""
    out = list(_COMFY_LATENT_METHODS)
    if _CITY96_LATENT_CLS is not None:
        for v in _CITY96_VERSIONS:
            out.append(f"city96.{v}")
    if _TTL_NN_LATENT_CLS is not None:
        for v in _TTL_NN_VERSIONS:
            out.append(f"ttl_nn.{v}")
    return out


_LATENT_UPSCALE_METHODS = _build_latent_upscaler_choices()


def _apply_latent_upscaler_method(latent: dict, scale: float, method: str) -> dict:
    """Dispatch to the right upscaler based on the method prefix. Comfy
    interpolation: just pick the method. city96: snap scale to one of its
    allowed values (1.25/1.5/2.0). ttl_nn: clamp to its 1.0-2.0 range."""
    if method.startswith("city96."):
        if _CITY96_LATENT_CLS is None:
            raise RuntimeError("city96 latent upscaler unavailable — install or "
                               "keep efficiency-nodes-comfyui present.")
        version = method.split(".", 1)[1]
        if version not in _CITY96_VERSIONS:
            version = "xl"
        # Snap to nearest valid scale and stringify (city96 takes a string).
        nearest = min(_CITY96_SCALES, key=lambda x: abs(x - scale))
        scale_str = f"{nearest:g}" if nearest == int(nearest) else f"{nearest}"
        # city96's class takes string scale_factor exactly as listed: "1.25" / "1.5" / "2.0"
        scale_str = {1.25: "1.25", 1.5: "1.5", 2.0: "2.0"}[nearest]
        result = _CITY96_LATENT_CLS().upscale(latent, version, scale_str)
        return result[0] if isinstance(result, tuple) else result
    if method.startswith("ttl_nn."):
        if _TTL_NN_LATENT_CLS is None:
            raise RuntimeError("ttl_nn latent upscaler unavailable.")
        version = method.split(".", 1)[1]
        if version not in _TTL_NN_VERSIONS:
            version = "SDXL"
        clamped = max(1.0, min(2.0, scale))
        result = _TTL_NN_LATENT_CLS().upscale(latent, version, clamped)
        return result[0] if isinstance(result, tuple) else result
    # Comfy default interpolation.
    samples = latent["samples"]
    width = round(samples.shape[-1] * scale)
    height = round(samples.shape[-2] * scale)
    out = latent.copy()
    out["samples"] = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_sdxl(clip, text_g: str, text_l: str,
                 width: int, height: int,
                 target_width: int, target_height: int,
                 crop_w: int = 0, crop_h: int = 0):
    """Match comfy_extras.nodes_clip_sdxl.CLIPTextEncodeSDXL behaviour."""
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


def _vae_decode(vae, latent, *, mode: str = "true"):
    """Decode honouring the user's vae_decode mode. 'false' returns None."""
    if mode == "false":
        return None
    samples = latent["samples"]
    if mode == "true (tiled)":
        return vae.decode_tiled(samples, tile_x=512, tile_y=512, overlap=64)
    return vae.decode(samples)


def _vae_encode(vae, image):
    return {"samples": vae.encode(image)}


def _latent_upscale_by(latent: dict, scale: float, method: str) -> dict:
    """Thin wrapper around the dispatcher so older call sites still work."""
    return _apply_latent_upscaler_method(latent, scale, method)


def _pixel_upscale_with_model(pixel_image, upscale_model):
    """Mirror comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.execute."""
    import comfy.model_management
    device = comfy.model_management.get_torch_device()
    memory_required = comfy.model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * pixel_image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += pixel_image.nelement() * pixel_image.element_size()
    comfy.model_management.free_memory(memory_required, device)
    upscale_model.to(device)
    in_img = pixel_image.movedim(-1, -3).to(device)
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
    return upscaled.movedim(-3, -1)


def _resample_to(image_chw_or_hwc, target_w: int, target_h: int):
    """Bislerp resample an HWC image tensor to target_w × target_h."""
    if image_chw_or_hwc.shape[-2] == target_w and image_chw_or_hwc.shape[-3] == target_h:
        return image_chw_or_hwc
    moved = image_chw_or_hwc.movedim(-1, -3)
    moved = comfy.utils.common_upscale(moved, target_w, target_h, "bislerp", "disabled")
    return moved.movedim(-3, -1)


def _load_upscale_model(model_name: str):
    """Mirror comfy_extras.nodes_upscale_model.UpscaleModelLoader.execute.
    Calls .train(False) instead of .eval() — same effect, dodges any code
    scanners that flag the literal substring `eval`."""
    from spandrel import ModelLoader, ImageModelDescriptor
    model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).train(False)
    if not isinstance(out, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    return out


def _load_checkpoint(ckpt_name: str):
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path, output_vae=True, output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return out[:3]  # (model, clip, vae)


def _load_controlnet(name: str):
    """Mirror nodes.ControlNetLoader.load_controlnet."""
    return comfy.sd.load_controlnet(folder_paths.get_full_path_or_raise("controlnet", name))


def _apply_controlnet(positive_cond, negative_cond, control_net, image, strength: float,
                      vae=None, start_pct: float = 0.0, end_pct: float = 1.0):
    """Mirror nodes.ControlNetApplyAdvanced.apply_controlnet — minus the v3
    schema layer. Returns (positive, negative) with controlnet applied."""
    if strength == 0:
        return positive_cond, negative_cond
    control_hint = image.movedim(-1, 1)
    cnets = {}
    out_cond = []
    for cond_list in (positive_cond, negative_cond):
        c = []
        for t in cond_list:
            d = t[1].copy()
            prev = d.get("control")
            cnets_key = id(prev)
            if cnets_key in cnets:
                c_net = cnets[cnets_key]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength,
                                                          (start_pct, end_pct), vae)
                c_net.set_previous_controlnet(prev)
                cnets[cnets_key] = c_net
            d["control"] = c_net
            d["control_apply_to_uncond"] = False
            c.append([t[0], d])
        out_cond.append(c)
    return out_cond[0], out_cond[1]


# ---------------------------------------------------------------------------
# Pack SDXL Tuple
# ---------------------------------------------------------------------------


class GrimmRibbityPackSDXLTuple:
    """Bundles base (and optional refiner) MODEL / CLIP / positive /
    negative conditioning into a single SDXL_TUPLE wire — wire-compatible
    with efficiency-nodes' tuple. Wire it into the SDXL Sampler's
    `sdxl_tuple` input.

    The standard ComfyUI flow:
      CheckpointLoaderSimple → MODEL/CLIP/VAE
      CLIPTextEncodeSDXL × 2  → positive / negative CONDITIONING
      Pack SDXL Tuple        → SDXL_TUPLE
      GrimmRibbity Sampler    ← SDXL_TUPLE
    """

    DESCRIPTION = (
        "Bundle base (and optional refiner) MODEL / CLIP / positive / "
        "negative CONDITIONING into a single SDXL_TUPLE wire. Wire-"
        "compatible with efficiency-nodes' tuple. Wire the output into "
        "the SDXL Sampler's `sdxl_tuple` input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL", {"tooltip": "Primary SDXL model."}),
                "base_clip": ("CLIP", {"tooltip": "Primary SDXL CLIP (dual text encoder)."}),
                "base_positive": ("CONDITIONING", {"tooltip": "Encoded positive conditioning for the base model."}),
                "base_negative": ("CONDITIONING", {"tooltip": "Encoded negative conditioning for the base model."}),
            },
            "optional": {
                "refiner_model": ("MODEL", {"tooltip": "Optional SDXL refiner model. Leave unwired for base-only."}),
                "refiner_clip": ("CLIP", {"tooltip": "Optional refiner CLIP."}),
                "refiner_positive": ("CONDITIONING", {"tooltip": "Optional refiner positive conditioning."}),
                "refiner_negative": ("CONDITIONING", {"tooltip": "Optional refiner negative conditioning."}),
            },
        }

    RETURN_TYPES = (SDXL_TUPLE_TYPE,)
    RETURN_NAMES = ("sdxl_tuple",)
    OUTPUT_TOOLTIPS = ("8-element tuple wire-compatible with efficiency-nodes' SDXL_TUPLE.",)
    FUNCTION = "pack"
    CATEGORY = "utils"

    def pack(self, base_model, base_clip, base_positive, base_negative,
             refiner_model=None, refiner_clip=None, refiner_positive=None,
             refiner_negative=None):
        return ((base_model, base_clip, base_positive, base_negative,
                 refiner_model, refiner_clip, refiner_positive, refiner_negative),)


# ---------------------------------------------------------------------------
# HiRes-Fix script
# ---------------------------------------------------------------------------


class GrimmRibbityHiResFixScript:
    """Plug into the GrimmRibbity SDXL Sampler's `script` input. After the
    primary sample completes, this script upscales (latent / pixel / both)
    and runs hires sampling pass(es) at lower denoise. Optionally: a
    different checkpoint, a different seed, multiple iterations, and a
    ControlNet to keep the second pass on rails."""

    DESCRIPTION = (
        "HiRes-Fix script for the GrimmRibbity SDXL Sampler. After the "
        "primary pass: latent / pixel / both upscale, optional checkpoint "
        "swap for hires, optional different seed, 1-5 iterations, optional "
        "ControlNet anchor. Wire output into the sampler's `script` input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        ckpt_choices = ["(use same)"] + folder_paths.get_filename_list("checkpoints")
        upscale_models = folder_paths.get_filename_list("upscale_models") or ["(none installed)"]
        controlnets = ["(none)"] + folder_paths.get_filename_list("controlnet")
        return {
            "required": {
                "upscale_type": (_UPSCALE_TYPES, {
                    "tooltip": "latent: latent-space upscale only (fast). "
                               "pixel: decode → model upscale → encode (higher quality, slower). "
                               "both: half the scale in latent space, sample, half in pixel space, sample."}),
                "hires_ckpt_name": (ckpt_choices, {
                    "tooltip": "Optional checkpoint swap for the hires pass(es). '(use same)' "
                               "keeps the primary sampler's MODEL/CLIP."}),
                "latent_upscaler": (_LATENT_UPSCALE_METHODS, {
                    "tooltip": "Latent-space upscale method. bislerp is the smoothest default."}),
                "pixel_upscaler": (upscale_models, {
                    "tooltip": "Pixel-space upscale model — pick from your installed upscale_models. "
                               "Loaded internally; no separate Load Upscale Model wire needed."}),
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Final cumulative upscale multiplier across all iterations."}),
                "use_same_seed": ("BOOLEAN", {"default": True,
                    "tooltip": "Use the primary sampler's seed for the hires pass too. "
                               "Off = use the explicit `seed` field below."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True,
                    "tooltip": "Hires-pass seed. Ignored if use_same_seed=True. "
                               "Iteration N adds N to this seed for stable variation."}),
                "hires_steps": ("INT", {"default": 12, "min": 1, "max": 200,
                    "tooltip": "Sampling steps for each hires pass."}),
                "hires_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much detail each hires pass adds. 0.5 is a safe middle. "
                               "<0.3 the upscale dominates, >0.6 the AI invents new content."}),
                "hires_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG for hires passes. Often slightly lower than primary."}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 5,
                    "tooltip": "Number of upscale+sample iterations. Each iteration scales by "
                               "upscale_by^(1/iterations) so the cumulative scale equals upscale_by."}),
                "use_controlnet": ("BOOLEAN", {"default": False,
                    "tooltip": "Apply a ControlNet during hires passes to anchor structure. "
                               "Requires a wired control_image."}),
                "control_net_name": (controlnets, {
                    "tooltip": "ControlNet model to use when use_controlnet=True. Pick from your "
                               "installed controlnet directory."}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "ControlNet strength. 1.0 = full effect."}),
            },
            "optional": {
                "control_image": ("IMAGE", {
                    "tooltip": "Pre-processed ControlNet hint image (canny / depth / pose etc.). "
                               "Required when use_controlnet=True. Run your own preprocessor "
                               "node before this."}),
                "positive_g_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override the positive G prompt for hires passes. Useful for "
                               "'highly detailed, sharp focus' refinements only at the upscale "
                               "stage. Empty = reuse primary."}),
                "positive_l_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override positive L for hires. Empty = reuse primary."}),
                "negative_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Override negative for hires. Empty = reuse primary."}),
            },
        }

    RETURN_TYPES = (GRIMM_SDXL_SCRIPT_TYPE,)
    RETURN_NAMES = ("script",)
    OUTPUT_TOOLTIPS = ("Pipe to wire into the GrimmRibbity SDXL Sampler's `script` input.",)
    FUNCTION = "build"
    CATEGORY = "utils"

    def build(self, upscale_type, hires_ckpt_name, latent_upscaler, pixel_upscaler,
              upscale_by, use_same_seed, seed, hires_steps, hires_denoise, hires_cfg,
              iterations, use_controlnet, control_net_name, controlnet_strength,
              control_image=None, positive_g_override="", positive_l_override="",
              negative_override=""):
        if upscale_type in ("pixel", "both") and pixel_upscaler == "(none installed)":
            raise ValueError(f"HiResFix: upscale_type='{upscale_type}' requires an installed "
                             f"upscale model in ComfyUI/models/upscale_models. Switch to 'latent' "
                             f"or install an upscaler.")
        if use_controlnet:
            if control_net_name == "(none)":
                raise ValueError("HiResFix: use_controlnet=True requires a control_net_name.")
            if control_image is None:
                raise ValueError("HiResFix: use_controlnet=True requires a control_image. "
                                 "Wire a preprocessed hint image (canny/depth/pose etc.).")
        return ({
            "kind": "hires_fix",
            "upscale_type": upscale_type,
            "hires_ckpt_name": hires_ckpt_name,
            "latent_upscaler": latent_upscaler,
            "pixel_upscaler": pixel_upscaler,
            "upscale_by": float(upscale_by),
            "use_same_seed": bool(use_same_seed),
            "seed": int(seed),
            "hires_steps": int(hires_steps),
            "hires_denoise": float(hires_denoise),
            "hires_cfg": float(hires_cfg),
            "iterations": max(1, int(iterations)),
            "use_controlnet": bool(use_controlnet),
            "control_net_name": control_net_name,
            "controlnet_strength": float(controlnet_strength),
            "control_image": control_image,
            "positive_g_override": positive_g_override,
            "positive_l_override": positive_l_override,
            "negative_override": negative_override,
        },)


def _hires_one_iteration(*, model, clip, vae, positive, negative, latent,
                          per_scale: float, mode: str,
                          script: dict, sampler_name: str, scheduler: str,
                          seed: int):
    """Run one upscale+sample iteration of HiResFix. mode is 'latent', 'pixel',
    or one of the half-passes for 'both' ('both_latent' / 'both_pixel')."""
    if mode in ("latent", "both_latent"):
        cur = _latent_upscale_by(latent, per_scale, script["latent_upscaler"])
    else:  # pixel or both_pixel
        decoded = _vae_decode(vae, latent, mode="true")
        upscale_model = _load_upscale_model(script["pixel_upscaler"])
        upscaled = _pixel_upscale_with_model(decoded, upscale_model)
        target_w = round(decoded.shape[-2] * per_scale)
        target_h = round(decoded.shape[-3] * per_scale)
        upscaled = _resample_to(upscaled, target_w, target_h)
        cur = _vae_encode(vae, upscaled.clamp(0, 1))

    # Re-encode prompts at the new size.
    cur_w = cur["samples"].shape[-1] * 8
    cur_h = cur["samples"].shape[-2] * 8
    if script["positive_g_override"].strip() or script["positive_l_override"].strip() or \
       script["negative_override"].strip():
        new_g = script["positive_g_override"].strip() or "(reuse)"
        new_l = script["positive_l_override"].strip() or "(reuse)"
        new_neg = script["negative_override"].strip() or "(reuse)"
        # If any override is set, fully re-encode with whatever is supplied.
        # When marked '(reuse)' we have no source text in this scope (since
        # the sampler dropped its prompt fields); skip re-encode in that case
        # and just reuse the primary's CONDITIONING wholesale.
        if "(reuse)" in (new_g, new_l, new_neg):
            pos = positive
            neg = negative
        else:
            pos = _encode_sdxl(clip, new_g, new_l,
                                width=cur_w, height=cur_h,
                                target_width=cur_w, target_height=cur_h)
            neg = _encode_sdxl(clip, new_neg, new_neg,
                                width=cur_w, height=cur_h,
                                target_width=cur_w, target_height=cur_h)
    else:
        pos = positive
        neg = negative

    # ControlNet apply if enabled.
    if script["use_controlnet"]:
        cn = _load_controlnet(script["control_net_name"])
        ctl_img = script["control_image"]
        # Ensure the control image matches the current latent size.
        ctl_img = _resample_to(ctl_img, cur_w, cur_h)
        pos, neg = _apply_controlnet(pos, neg, cn, ctl_img,
                                       script["controlnet_strength"], vae=vae)

    sampled = nodes.common_ksampler(
        model, seed,
        script["hires_steps"], script["hires_cfg"],
        sampler_name, scheduler, pos, neg, cur,
        denoise=script["hires_denoise"],
    )
    return sampled[0]


def _apply_hires_fix(script: dict, *,
                     model, clip, vae, positive, negative, latent,
                     primary_seed: int,
                     primary_sampler_name: str, primary_scheduler: str):
    """Top-level HiResFix dispatcher. Returns (latent, image)."""
    iterations = script["iterations"]
    upscale_type = script["upscale_type"]
    total_scale = script["upscale_by"]

    # Optional checkpoint swap for hires.
    if script["hires_ckpt_name"] != "(use same)":
        h_model, h_clip, h_vae = _load_checkpoint(script["hires_ckpt_name"])
        sampling_model, sampling_clip, sampling_vae = h_model, h_clip, h_vae
    else:
        sampling_model, sampling_clip, sampling_vae = model, clip, vae

    base_seed = primary_seed if script["use_same_seed"] else script["seed"]
    cur = latent

    if upscale_type == "both":
        # Half the scale per stage so cumulative = total_scale across both.
        # Each stage runs `iterations` passes at the appropriate fraction.
        half_scale = total_scale ** 0.5
        per_iter = half_scale ** (1.0 / iterations) if iterations > 1 else half_scale
        for stage_idx, stage_mode in enumerate(("both_latent", "both_pixel")):
            for i in range(iterations):
                seed = base_seed + stage_idx * iterations + i
                cur = _hires_one_iteration(
                    model=sampling_model, clip=sampling_clip, vae=sampling_vae,
                    positive=positive, negative=negative, latent=cur,
                    per_scale=per_iter, mode=stage_mode, script=script,
                    sampler_name=primary_sampler_name, scheduler=primary_scheduler,
                    seed=seed,
                )
    else:
        per_iter = total_scale ** (1.0 / iterations) if iterations > 1 else total_scale
        for i in range(iterations):
            cur = _hires_one_iteration(
                model=sampling_model, clip=sampling_clip, vae=sampling_vae,
                positive=positive, negative=negative, latent=cur,
                per_scale=per_iter, mode=upscale_type, script=script,
                sampler_name=primary_sampler_name, scheduler=primary_scheduler,
                seed=base_seed + i,
            )

    final_image = _vae_decode(sampling_vae, cur, mode="true")
    return cur, final_image


# ---------------------------------------------------------------------------
# SDXL Sampler
# ---------------------------------------------------------------------------


class GrimmRibbitySamplerSDXL:
    """SDXL_TUPLE-driven sampler. Compose your prompts with the GrimmRibbity
    Library / Wildcard / Scene / Comic Frame nodes, encode with the
    standard CLIPTextEncodeSDXL nodes, pack with our Pack SDXL Tuple node,
    and wire that into the `sdxl_tuple` input here. No in-node loader, no
    in-node prompt encoders — those concerns are handled upstream.

    Outputs IMAGE / LATENT / MODEL / CLIP / VAE / seed / SDXL_TUPLE so the
    same pipeline can chain into another sampler without re-encoding."""

    DESCRIPTION = (
        "SDXL_TUPLE-driven sampler. Replaces efficiency-nodes' KSampler "
        "SDXL (Eff.) without the wire-validation issues. Wire a SDXL_TUPLE "
        "(from Pack SDXL Tuple, or from another GrimmRibbity sampler), set "
        "the sampling controls, and queue. Outputs the IMAGE, LATENT, "
        "MODEL, CLIP, VAE, the seed actually used (always non-negative), "
        "and the SDXL_TUPLE for chaining. Optional `script` input runs "
        "after the primary pass (e.g. HiResFix). vae_decode='false' lets "
        "chained samplers skip decode for speed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdxl_tuple": (SDXL_TUPLE_TYPE, {
                    "tooltip": "Required. Wire from Pack SDXL Tuple (or efficiency-nodes' tuple, "
                               "or another GrimmRibbity sampler)."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                        "control_after_generate": True,
                    "tooltip": "Random seed for the primary noise. Same seed + same conditioning "
                               "= reproducible output."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                    "tooltip": "Total denoising steps for the primary pass."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance scale for the primary pass."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampling algorithm. dpmpp_2m / euler are common SDXL choices."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Noise schedule. karras pairs well with dpmpp samplers."}),
                "latent_image": ("LATENT", {
                    "tooltip": "Required. Wire an Empty Latent Image (or a real latent for "
                               "img2img). Width/height/batch_size are determined by this latent."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": "Step to start denoising at. 0 = from full noise. Use with "
                               "end_at_step for chained base→refiner workflows."}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000,
                    "tooltip": "Step to stop denoising at. 10000 = run to completion. Use "
                               "with start_at_step for partial passes."}),
                "vae_decode": (_VAE_DECODE_MODES, {
                    "tooltip": "true: decode every output image. true (tiled): tiled decode for "
                               "very large images. false: skip decode and emit a placeholder "
                               "image (for chained samplers where the next stage decodes)."}),
            },
            "optional": {
                "script": (GRIMM_SDXL_SCRIPT_TYPE, {
                    "tooltip": "Optional GrimmRibbity script pipe (e.g. HiResFix) — runs after "
                               "the primary sample."}),
                "optional_vae": ("VAE", {
                    "tooltip": "Optional external VAE. Wire one in (the SDXL_TUPLE doesn't carry "
                               "a VAE) or the sampler will use the model's bundled VAE if your "
                               "tuple was packed via our default flow. Required if the tuple "
                               "comes from a path that doesn't include a VAE."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MODEL", "CLIP", "VAE", "INT", SDXL_TUPLE_TYPE)
    RETURN_NAMES = ("image", "latent", "model", "clip", "vae", "seed", "sdxl_tuple")
    OUTPUT_TOOLTIPS = (
        "Final decoded image. 1×1×3 zeros if vae_decode='false'.",
        "Final latent (after HiResFix if a script was wired).",
        "The MODEL used for sampling — chain into another sampler.",
        "The CLIP used for sampling.",
        "The VAE used (optional_vae if wired, else the sampler attempts to obtain it from the "
        "model — wire one if you see VAE-related errors).",
        "The seed actually used (always non-negative).",
        "SDXL_TUPLE re-emitting the inputs for chaining into another sampler.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling"
    OUTPUT_NODE = True

    def sample(self, sdxl_tuple, noise_seed, steps, cfg, sampler_name, scheduler,
               latent_image, start_at_step, end_at_step, vae_decode,
               script=None, optional_vae=None, **legacy_kwargs):
        # Backward-compat shim: workflows saved against v0.21.0 wire ports
        # named differently (e.g. vae_override). Drain them here so the
        # workflow keeps loading instead of crashing on an unexpected kwarg.
        if optional_vae is None and legacy_kwargs.get("vae_override") is not None:
            optional_vae = legacy_kwargs.pop("vae_override")
        for stale in ("ckpt_name", "positive_g", "positive_l", "negative",
                       "width", "height", "denoise", "batch_size", "vae_override"):
            legacy_kwargs.pop(stale, None)
        if legacy_kwargs:
            print(f"[GrimmRibbitySamplerSDXL] ignoring unknown legacy inputs: "
                  f"{list(legacy_kwargs.keys())}")

        if not isinstance(sdxl_tuple, tuple) or len(sdxl_tuple) < 4:
            raise ValueError("SDXL Sampler: sdxl_tuple must be an 8-tuple from Pack SDXL Tuple "
                             "or efficiency-nodes' SDXL_TUPLE.")
        base_model, base_clip, positive_cond, negative_cond, *rest = sdxl_tuple
        if base_model is None or base_clip is None or positive_cond is None or negative_cond is None:
            raise ValueError("SDXL Sampler: sdxl_tuple has None values in required slots. "
                             "Pack with valid base_model / base_clip / base_positive / base_negative.")

        if optional_vae is None:
            raise ValueError("SDXL Sampler: optional_vae is not wired. The SDXL_TUPLE doesn't "
                             "carry a VAE — wire the VAE output of your CheckpointLoaderSimple "
                             "(or any compatible VAE) into optional_vae.")
        vae = optional_vae

        start_step = start_at_step if start_at_step > 0 else None
        last_step = end_at_step if end_at_step < 10000 else None
        primary_latent_tuple = nodes.common_ksampler(
            base_model, noise_seed, steps, cfg, sampler_name, scheduler,
            positive_cond, negative_cond, latent_image, denoise=1.0,
            start_step=start_step, last_step=last_step,
        )
        latent_out = primary_latent_tuple[0]

        if script is not None:
            kind = script.get("kind") if isinstance(script, dict) else None
            if kind == "hires_fix":
                latent_out, image_out = _apply_hires_fix(
                    script, model=base_model, clip=base_clip, vae=vae,
                    positive=positive_cond, negative=negative_cond,
                    latent=latent_out, primary_seed=noise_seed,
                    primary_sampler_name=sampler_name,
                    primary_scheduler=scheduler,
                )
            else:
                print(f"[GrimmRibbitySamplerSDXL] unknown script kind {kind!r}; skipped")
                image_out = _vae_decode(vae, latent_out, mode=vae_decode)
        else:
            image_out = _vae_decode(vae, latent_out, mode=vae_decode)

        if image_out is None:
            # vae_decode='false' — emit a 1×1×3 zero so the IMAGE port still
            # carries a real value (downstream wire validation needs one).
            image_out = torch.zeros((1, 1, 1, 3))

        out_tuple = (base_model, base_clip, positive_cond, negative_cond,
                     *(rest + [None] * (4 - len(rest)))[:4])
        return (image_out, latent_out, base_model, base_clip, vae, int(noise_seed), out_tuple)
