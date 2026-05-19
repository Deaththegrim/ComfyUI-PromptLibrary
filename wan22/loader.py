"""Wan2.2 i2v 14B combined loader.

Wan2.2-A14B ships as two diffusion_models files (HighNoise + LowNoise experts).
This node loads both UNETs, the Wan2.2 VAE, and the UMT5-XXL text encoder in
one block so the workflow only has one source-of-truth node for the model
stack.

Supports BOTH safetensors (via comfy.sd.load_diffusion_model) and GGUF quants
(via the ComfyUI-GGUF custom node pack). Dispatch is by file extension. The
dropdown is the union of diffusion_models/*.safetensors and unet_gguf/*.gguf
so the user just picks the file, regardless of format.
"""

from __future__ import annotations

import logging
import os

import torch

import comfy.sd
import comfy.utils
import folder_paths


_log = logging.getLogger(__name__)
_WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _dtype_options(weight_dtype: str) -> dict:
    opts: dict = {}
    if weight_dtype == "fp8_e4m3fn":
        opts["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        opts["dtype"] = torch.float8_e4m3fn
        opts["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        opts["dtype"] = torch.float8_e5m2
    return opts


def _gguf_names() -> list[str]:
    # ComfyUI-GGUF registers the "unet_gguf" folder when it loads. If it's
    # not installed, just return an empty list — the dropdown then only
    # offers safetensors.
    try:
        return list(folder_paths.get_filename_list("unet_gguf"))
    except Exception:
        return []


def _unet_choices() -> list[str]:
    # Tag GGUF entries so the dispatcher can tell them apart from a
    # same-named safetensors file (rare but possible). The tag is a visible
    # suffix — users see "wan2.2…Q4_K_S.gguf  [gguf]" in the dropdown.
    safetensors = list(folder_paths.get_filename_list("diffusion_models"))
    gguf = _gguf_names()
    return safetensors + [f"{name}  [gguf]" for name in gguf]


def _load_unet_dispatch(choice: str, model_options: dict):
    """Load one expert from either safetensors or GGUF, returning a MODEL."""
    if choice.endswith("  [gguf]"):
        unet_name = choice[: -len("  [gguf]")]
        return _load_unet_gguf(unet_name)
    # Plain safetensors path.
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", choice)
    return comfy.sd.load_diffusion_model(unet_path, model_options=model_options)


def _load_unet_gguf(unet_name: str):
    """Delegate to ComfyUI-GGUF's UnetLoaderGGUF if available."""
    try:
        # The custom-node pack folder name has a dash; import via the
        # registered top-level module name it sets up at install time.
        # ComfyUI mounts each custom_nodes/<name> as a module by its
        # directory basename, so we try a few likely import paths.
        unet_loader_cls = _resolve_gguf_loader()
    except ImportError as e:
        raise RuntimeError(
            "GGUF file selected but ComfyUI-GGUF is not importable. "
            "Install it via Manager or git-clone Calcium-Crystal/ComfyUI-GGUF "
            "into custom_nodes/."
        ) from e
    return unet_loader_cls().load_unet(unet_name)[0]


def _resolve_gguf_loader():
    # Try the common module names ComfyUI-GGUF exposes itself under. The
    # repo's __init__.py registers NODE_CLASS_MAPPINGS but ComfyUI imports
    # the package using a sanitized directory name, so we walk a small
    # list rather than guess one path.
    import importlib
    for name in ("ComfyUI_GGUF", "ComfyUI-GGUF", "comfyui_gguf"):
        try:
            mod = importlib.import_module(f"{name}.nodes")
            return mod.UnetLoaderGGUF
        except ImportError:
            continue
    # Last resort: load the file directly from its known install path.
    import importlib.util
    pack_path = os.path.join(
        folder_paths.base_path, "custom_nodes", "ComfyUI-GGUF", "nodes.py")
    if not os.path.isfile(pack_path):
        raise ImportError(f"ComfyUI-GGUF/nodes.py not found at {pack_path}")
    spec = importlib.util.spec_from_file_location(
        "_grimmribbity_gguf_nodes", pack_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UnetLoaderGGUF


class GrimmRibbityWan22I2VLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unets = _unet_choices()
        vaes = folder_paths.get_filename_list("vae")
        clips = folder_paths.get_filename_list("text_encoders")
        return {
            "required": {
                "high_noise_unet": (unets, {
                    "tooltip": "HighNoise expert checkpoint (early timesteps). "
                               ".gguf files are auto-detected and routed through "
                               "ComfyUI-GGUF; safetensors use the core loader."}),
                "low_noise_unet": (unets, {
                    "tooltip": "LowNoise expert checkpoint (late timesteps)."}),
                "vae_name": (vaes, {
                    "tooltip": "Wan2.2 VAE (wan2.2_vae.safetensors) — also accepts "
                               "the Wan2.1 VAE (wan_2.1_vae.safetensors), which "
                               "remains compatible with 2.2 14B."}),
                "clip_name": (clips, {
                    "tooltip": "UMT5-XXL text encoder. Loaded as CLIPType.WAN."}),
                "weight_dtype": (_WEIGHT_DTYPES, {
                    "tooltip": "Override dtype for safetensors UNETs. Ignored "
                               "for .gguf files (their quant level is fixed)."}),
            },
            "optional": {
                "clip_device": (["default", "cpu"], {"advanced": True,
                    "tooltip": "Force the text encoder onto CPU (frees VRAM at "
                               "the cost of a one-time encode latency)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "VAE", "CLIP")
    RETURN_NAMES = ("model_high", "model_low", "vae", "clip")
    OUTPUT_TOOLTIPS = (
        "HighNoise expert — feeds the first stage of the two-stage sampler.",
        "LowNoise expert — feeds the second stage.",
        "Wan2.2 VAE for encoding the start image and decoding the final latent.",
        "UMT5-XXL CLIP for positive/negative text encoding.",
    )
    FUNCTION = "load"
    CATEGORY = "GrimmRibbity/Wan2.2"
    DESCRIPTION = (
        "One-stop loader for the Wan2.2-A14B i2v stack: HighNoise UNET + "
        "LowNoise UNET + Wan2.2 VAE + UMT5-XXL CLIP. Handles both safetensors "
        "and GGUF formats. Feed both MODEL outputs into the GrimmRibbity "
        "Two-Stage KSampler; run the start image through core WanImageToVideo "
        "/ WanFirstLastFrameToVideo / Wan22ImageToVideoLatent for i2v "
        "conditioning."
    )

    def load(self, high_noise_unet, low_noise_unet, vae_name, clip_name,
             weight_dtype, clip_device="default"):
        model_options = _dtype_options(weight_dtype)
        model_high = _load_unet_dispatch(high_noise_unet, model_options)
        model_low = _load_unet_dispatch(low_noise_unet, model_options)

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)
        vae.throw_exception_if_invalid()

        clip_options: dict = {}
        if clip_device == "cpu":
            clip_options["load_device"] = torch.device("cpu")
            clip_options["offload_device"] = torch.device("cpu")
        clip_path = folder_paths.get_full_path_or_raise(
            "text_encoders", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.WAN,
            model_options=clip_options,
        )

        return (model_high, model_low, vae, clip)
