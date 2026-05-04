"""Civitai-compatible Save Image node.

ComfyUI's built-in SaveImage doesn't write A1111-style `parameters` PNG chunks,
so when you upload to Civitai it falls back to filename guessing and frequently
mis-attributes resources. This node:

- Lets you pick the checkpoint from any of `checkpoints/`, `diffusion_models/`,
  or `unet/` (the existing third-party Civitai-metadata nodes only look in
  `checkpoints/`, which mis-flags Anima models that live in `diffusion_models/`).
- Computes SHA256 (cached on disk) for the chosen checkpoint and any optional
  LoRAs, and embeds them in both the `Model hash:` field and the `Hashes:` JSON
  block — the two places Civitai checks.
- Also writes ComfyUI's own `prompt` + `workflow` chunks so the saved image
  round-trips back into ComfyUI cleanly.

Hash cache lives at `data/hash_cache.json`, keyed by `folder/name|size|mtime`.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

try:
    import folder_paths
except ImportError:  # standalone import for tests
    folder_paths = None

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
HASH_CACHE_PATH = DATA_DIR / "hash_cache.json"

# The folders that hold full checkpoints / single-file models. `unet/` is the
# legacy name for what is now `diffusion_models/` on newer ComfyUIs; we list
# both because installs with --extra-model-paths can populate either.
_MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet")
_LORA_FOLDER = "loras"

_NONE_LABEL = "(none)"
_PREFIX_SEP = "::"  # "checkpoints::anima_v4.safetensors"

_cache_lock = threading.Lock()


# --------------------------------------------------------------------------
# Hash cache

def _load_hash_cache() -> dict:
    if not HASH_CACHE_PATH.exists():
        return {}
    try:
        with HASH_CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_hash_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HASH_CACHE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(HASH_CACHE_PATH)


def _file_signature(path: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{st.st_size}|{int(st.st_mtime)}"


def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cached_sha256(label: str, full_path: str) -> str | None:
    """SHA256 with on-disk cache. label is the dropdown entry (folder/name);
    cache key includes size+mtime so a swapped file invalidates automatically.
    """
    sig = _file_signature(full_path)
    if sig is None:
        return None
    key = f"{label}|{sig}"
    with _cache_lock:
        cache = _load_hash_cache()
        if key in cache:
            return cache[key]
    # Compute outside the lock — SHA256 of a 6 GB checkpoint takes a while.
    sha = _compute_sha256(full_path)
    with _cache_lock:
        cache = _load_hash_cache()
        cache[key] = sha
        _save_hash_cache(cache)
    return sha


# --------------------------------------------------------------------------
# Folder scanning

def _list_models_in_folder(folder: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return folder_paths.get_filename_list(folder)
    except Exception:
        return []


def list_all_model_choices() -> list[str]:
    """Flat dropdown list across the three model folders, prefixed so the
    folder is unambiguous when the same filename appears in two.

    `unet/` and `diffusion_models/` are aliases on modern ComfyUI — same
    physical files. We dedup by resolved full path so the dropdown only shows
    each model once, preferring the most descriptive folder name in the order
    listed in `_MODEL_FOLDERS`.
    """
    seen_paths: set[str] = set()
    seen_labels: set[str] = set()
    out: list[str] = [_NONE_LABEL]
    for folder in _MODEL_FOLDERS:
        for name in _list_models_in_folder(folder):
            full = None
            if folder_paths is not None:
                try:
                    full = folder_paths.get_full_path(folder, name)
                except Exception:
                    full = None
            key = os.path.realpath(full) if full else f"{folder}::{name}"
            if key in seen_paths:
                continue
            label = f"{folder}{_PREFIX_SEP}{name}"
            if label in seen_labels:
                continue
            seen_paths.add(key)
            seen_labels.add(label)
            out.append(label)
    return out


def list_lora_choices() -> list[str]:
    out = [_NONE_LABEL]
    out.extend(_list_models_in_folder(_LORA_FOLDER))
    return out


def resolve_model_path(label: str) -> tuple[str, str] | None:
    """Reverse a dropdown label into (folder, full_path). Returns None for the
    sentinel '(none)' or unknown labels."""
    if not label or label == _NONE_LABEL or _PREFIX_SEP not in label:
        return None
    folder, name = label.split(_PREFIX_SEP, 1)
    if folder_paths is None:
        return None
    try:
        full = folder_paths.get_full_path(folder, name)
    except Exception:
        return None
    if not full or not os.path.isfile(full):
        return None
    return name, full


def resolve_lora_path(name: str) -> str | None:
    if not name or name == _NONE_LABEL:
        return None
    if folder_paths is None:
        return None
    try:
        full = folder_paths.get_full_path(_LORA_FOLDER, name)
    except Exception:
        return None
    return full if full and os.path.isfile(full) else None


# --------------------------------------------------------------------------
# Metadata builder

def _strip_ext(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def build_a1111_parameters(
    *,
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int | None,
    sampler_name: str | None,
    scheduler: str | None,
    cfg: float | None,
    seed: int | None,
    model_name: str | None,
    model_sha256: str | None,
    loras: list[tuple[str, str | None, float]],  # (name, sha256, strength)
    extra: dict | None = None,
) -> str:
    """Format the A1111-style `parameters` string Civitai parses.

    Civitai matches the main checkpoint by AutoV2 hash (first 10 chars of
    SHA256), so we always emit `Model hash:` plus a `Hashes:` JSON block —
    the latter also carries LoRAs and any extras.
    """
    parts: list[str] = []
    parts.append((positive or "").strip())
    if negative and negative.strip():
        parts.append(f"Negative prompt: {negative.strip()}")

    fields: list[str] = []
    if steps is not None and steps > 0:
        fields.append(f"Steps: {int(steps)}")
    if sampler_name:
        sampler = sampler_name
        if scheduler and scheduler != "normal":
            sampler = f"{sampler_name} {scheduler}"
        fields.append(f"Sampler: {sampler}")
    if cfg is not None:
        fields.append(f"CFG scale: {cfg:g}")
    if seed is not None:
        fields.append(f"Seed: {int(seed)}")
    fields.append(f"Size: {int(width)}x{int(height)}")

    if model_sha256:
        fields.append(f"Model hash: {model_sha256[:10]}")
    if model_name:
        fields.append(f"Model: {_strip_ext(model_name)}")

    hashes: dict = {}
    if model_sha256:
        hashes["model"] = model_sha256[:10]
    for lname, lsha, _strength in loras:
        if not lsha:
            continue
        hashes[f"lora:{_strip_ext(lname)}"] = lsha[:10]
    if extra:
        for k, v in extra.items():
            hashes[k] = v
    if hashes:
        fields.append(f"Hashes: {json.dumps(hashes, separators=(',', ':'))}")

    # Reference each LoRA inline too — Civitai also picks these up via the
    # `<lora:name:strength>` syntax.
    if loras:
        lora_tokens = " ".join(f"<lora:{_strip_ext(n)}:{s}>" for n, _, s in loras)
        if lora_tokens:
            parts[0] = f"{parts[0]} {lora_tokens}".strip()

    if fields:
        parts.append(", ".join(fields))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Node

class CivitaiSaveImage:
    """Save IMAGE to disk with Civitai-compatible metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "GrimmRibbity"}),
                "model": (list_all_model_choices(),),
            },
            "optional": {
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "lora_1": (list_lora_choices(),),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (list_lora_choices(),),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_3": (list_lora_choices(),),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save(self, images, filename_prefix, model,
             positive="", negative="",
             lora_1=_NONE_LABEL, lora_1_strength=1.0,
             lora_2=_NONE_LABEL, lora_2_strength=1.0,
             lora_3=_NONE_LABEL, lora_3_strength=1.0,
             seed=0, steps=0, cfg=0.0,
             sampler_name="", scheduler="",
             prompt=None, extra_pnginfo=None):
        from PIL import Image, PngImagePlugin
        import numpy as np

        model_resolved = resolve_model_path(model)
        model_name = model_resolved[0] if model_resolved else None
        model_sha = get_cached_sha256(model, model_resolved[1]) if model_resolved else None

        loras: list[tuple[str, str | None, float]] = []
        for name, strength in (
            (lora_1, lora_1_strength),
            (lora_2, lora_2_strength),
            (lora_3, lora_3_strength),
        ):
            full = resolve_lora_path(name)
            if not full:
                continue
            loras.append((name, get_cached_sha256(name, full), float(strength)))

        output_dir = folder_paths.get_output_directory() if folder_paths else "output"
        full_prefix, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir,
                                              images.shape[2], images.shape[1])
            if folder_paths else (filename_prefix, filename_prefix, 0, "", "")
        )

        results = []
        for i, frame in enumerate(images):
            arr = frame
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            params = build_a1111_parameters(
                positive=positive,
                negative=negative,
                width=pil.width,
                height=pil.height,
                steps=steps if steps > 0 else None,
                sampler_name=sampler_name or None,
                scheduler=scheduler or None,
                cfg=cfg if cfg > 0 else None,
                seed=seed if seed > 0 else None,
                model_name=model_name,
                model_sha256=model_sha,
                loras=loras,
            )

            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("parameters", params)
            if prompt is not None:
                png_info.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    png_info.add_text(k, json.dumps(v))

            file_name = f"{filename}_{counter:05d}_.png"
            counter += 1
            full_path = os.path.join(full_prefix, file_name)
            pil.save(full_path, format="PNG", pnginfo=png_info, compress_level=4)
            results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": "output",
            })

        return {"ui": {"images": results}}
