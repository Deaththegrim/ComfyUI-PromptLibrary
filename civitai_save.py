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


# --------------------------------------------------------------------------
# Workflow introspection — read the PROMPT dict ComfyUI passes to the node and
# extract the model / LoRAs / KSampler params / prompt texts automatically.

_MODEL_LOADER_TYPES = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "CheckpointLoader": ("ckpt_name", "checkpoints"),
    "UNETLoader": ("unet_name", "diffusion_models"),
    "DiffusionModelLoader": ("model_name", "diffusion_models"),
}
_LORA_LOADER_TYPES = {
    "LoraLoader": ("lora_name", "strength_model"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
}
# rgthree's Power Lora Loader stores any number of LoRAs as `lora_1`, `lora_2`,
# … inputs whose values are dicts: {on, lora, strength, strengthTwo?}.
_RGTHREE_POWER_LORA_TYPE = "Power Lora Loader (rgthree)"
_KSAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustom",
                   "SamplerCustomAdvanced", "KSampler (Efficient)"}
_TEXT_ENCODE_TYPES = {"CLIPTextEncode", "CLIPTextEncodeSDXL", "BNK_CLIPTextEncodeAdvanced"}


def _link_source(value):
    """Extract the source node id from a connection link [node_id, output_idx]."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def _walk_model_chain(prompt: dict, start_node_id: str | None):
    """Walk a model-input link backwards through LoRA loaders to the actual
    checkpoint loader. Returns (model_label_or_None, [(lora_name, strength), ...]).

    `model_label_or_None` matches the format produced by `list_all_model_choices`
    (e.g. 'diffusion_models::anima_v4.safetensors') so it round-trips through
    `resolve_model_path`.
    """
    loras: list[tuple[str, float]] = []
    seen: set[str] = set()
    current = start_node_id
    while current and current in prompt and current not in seen:
        seen.add(current)
        node = prompt.get(current) or {}
        ctype = node.get("class_type")
        inputs = node.get("inputs") or {}

        if ctype in _LORA_LOADER_TYPES:
            name_key, strength_key = _LORA_LOADER_TYPES[ctype]
            lname = inputs.get(name_key)
            try:
                strength = float(inputs.get(strength_key, 1.0))
            except (TypeError, ValueError):
                strength = 1.0
            if isinstance(lname, str) and lname:
                loras.append((lname, strength))
            current = _link_source(inputs.get("model"))
            continue

        if ctype == _RGTHREE_POWER_LORA_TYPE:
            # `lora_1`, `lora_2`, … are applied in numeric order. We append
            # them in REVERSE so the outer `reversed(loras)` flips the whole
            # collection back into application order (matching the chain-of-
            # LoraLoader convention).
            slots = []
            for key, value in inputs.items():
                if not key.startswith("lora_") or not isinstance(value, dict):
                    continue
                try:
                    idx = int(key.split("_", 1)[1])
                except (ValueError, IndexError):
                    idx = 9999
                slots.append((idx, value))
            slots.sort(key=lambda s: s[0], reverse=True)
            for _, value in slots:
                if not value.get("on"):
                    continue
                lname = value.get("lora")
                try:
                    strength = float(value.get("strength", 1.0))
                except (TypeError, ValueError):
                    strength = 1.0
                if isinstance(lname, str) and lname and strength != 0:
                    loras.append((lname, strength))
            current = _link_source(inputs.get("model"))
            continue

        if ctype in _MODEL_LOADER_TYPES:
            name_key, folder = _MODEL_LOADER_TYPES[ctype]
            mname = inputs.get(name_key)
            if isinstance(mname, str) and mname:
                # Reverse list order — we walked back from KSampler, so LoRAs
                # are in reverse application order.
                return f"{folder}{_PREFIX_SEP}{mname}", list(reversed(loras))
            return None, list(reversed(loras))

        # Unknown node in the chain — try its `model` input if it has one.
        current = _link_source(inputs.get("model"))
    return None, list(reversed(loras))


def _resolve_text_link(prompt: dict, link_value, depth: int = 0) -> str:
    """Follow a CLIPTextEncode link and return its `text` input. Tolerates one
    level of indirection (e.g. a primitive Text node feeding the encoder)."""
    if depth > 4:
        return ""
    src = _link_source(link_value)
    if not src or src not in prompt:
        return ""
    node = prompt[src] or {}
    inputs = node.get("inputs") or {}
    if node.get("class_type") in _TEXT_ENCODE_TYPES:
        text = inputs.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            return _resolve_text_link(prompt, text, depth + 1)
    # Plain string-output node — most have a `text` or `string` field.
    for key in ("text", "string", "value"):
        v = inputs.get(key)
        if isinstance(v, str):
            return v
    return ""


_SAMPLER_PARAM_KEYS = ("seed", "steps", "cfg", "sampler_name", "scheduler",
                        "noise_seed")


def _find_primary_sampler(prompt: dict) -> tuple[str | None, dict | None]:
    """Pick the KSampler whose output feeds into the SaveImage path. Heuristic:
    last sampler (highest node id) in the graph — works for typical workflows
    including HiResFix chains where the upscaling sampler is later."""
    candidates = [(nid, n) for nid, n in prompt.items()
                  if (n or {}).get("class_type") in _KSAMPLER_TYPES]
    if not candidates:
        return None, None
    # Sort by integer node id when possible, else lex; take the highest.
    def _sort_key(pair):
        nid, _ = pair
        try:
            return (0, int(nid))
        except ValueError:
            return (1, nid)
    candidates.sort(key=_sort_key)
    nid, node = candidates[-1]
    return nid, node


def extract_workflow_metadata(prompt: dict | None) -> dict:
    """Read the PROMPT dict and pull out the bits Civitai cares about.

    Best-effort: every field is optional and missing pieces are simply omitted
    from the rendered metadata."""
    if not isinstance(prompt, dict) or not prompt:
        return {}

    out: dict = {}

    sampler_id, sampler = _find_primary_sampler(prompt)
    if sampler is not None:
        s_in = sampler.get("inputs") or {}
        for key in _SAMPLER_PARAM_KEYS:
            if key in s_in and not isinstance(s_in[key], list):
                out[key] = s_in[key]
        # Walk the model chain back from this sampler.
        model_label, loras = _walk_model_chain(prompt, _link_source(s_in.get("model")))
        if model_label:
            out["model_label"] = model_label
        if loras:
            out["loras"] = loras
        # Resolve positive / negative prompt text via the linked encoders.
        pos = _resolve_text_link(prompt, s_in.get("positive"))
        neg = _resolve_text_link(prompt, s_in.get("negative"))
        if pos:
            out["positive"] = pos
        if neg:
            out["negative"] = neg

    # Fallback: if no sampler found, still try to find a model loader.
    if "model_label" not in out:
        for nid, n in prompt.items():
            ctype = (n or {}).get("class_type")
            if ctype in _MODEL_LOADER_TYPES:
                name_key, folder = _MODEL_LOADER_TYPES[ctype]
                mname = (n.get("inputs") or {}).get(name_key)
                if isinstance(mname, str) and mname:
                    out["model_label"] = f"{folder}{_PREFIX_SEP}{mname}"
                    break

    # Normalise types — KSampler ints are sometimes floats from string sources.
    if "seed" in out and not isinstance(out["seed"], int):
        try:
            out["seed"] = int(out["seed"])
        except (TypeError, ValueError):
            out.pop("seed", None)
    if "noise_seed" in out and "seed" not in out:
        try:
            out["seed"] = int(out["noise_seed"])
        except (TypeError, ValueError):
            pass
        out.pop("noise_seed", None)
    if "steps" in out and not isinstance(out["steps"], int):
        try:
            out["steps"] = int(out["steps"])
        except (TypeError, ValueError):
            out.pop("steps", None)
    if "cfg" in out:
        try:
            out["cfg"] = float(out["cfg"])
        except (TypeError, ValueError):
            out.pop("cfg", None)

    return out


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

_AUTO_LABEL = "(auto-detect from workflow)"


class CivitaiSaveImage:
    """Drop-in replacement for SaveImage that writes Civitai-compatible
    metadata. Everything is auto-detected from the running workflow — you
    only need to wire the IMAGE input. Override any field if the auto pick
    is wrong (e.g., your workflow has two KSamplers and we picked the wrong
    one for the final pass)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "GrimmRibbity"}),
            },
            "optional": {
                "model_override": ([_AUTO_LABEL] + list_all_model_choices()[1:],),
                "positive_override": ("STRING", {"default": "", "multiline": True,
                                                  "placeholder": "leave blank to auto-detect"}),
                "negative_override": ("STRING", {"default": "", "multiline": True,
                                                  "placeholder": "leave blank to auto-detect"}),
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

    def save(self, images, filename_prefix,
             model_override=_AUTO_LABEL,
             positive_override="", negative_override="",
             prompt=None, extra_pnginfo=None):
        from PIL import Image, PngImagePlugin
        import numpy as np

        meta = extract_workflow_metadata(prompt)

        # Model: explicit override wins, else auto-detected, else empty.
        model_label = (
            model_override if (model_override and model_override != _AUTO_LABEL)
            else meta.get("model_label")
        )
        model_resolved = resolve_model_path(model_label) if model_label else None
        model_name = model_resolved[0] if model_resolved else None
        model_sha = (get_cached_sha256(model_label, model_resolved[1])
                     if model_resolved else None)

        # LoRAs come straight from the workflow trace.
        loras: list[tuple[str, str | None, float]] = []
        for lname, strength in meta.get("loras", []):
            full = resolve_lora_path(lname)
            if not full:
                continue
            loras.append((lname, get_cached_sha256(lname, full), float(strength)))

        positive = positive_override.strip() or meta.get("positive", "")
        negative = negative_override.strip() or meta.get("negative", "")
        seed = meta.get("seed")
        steps = meta.get("steps")
        cfg = meta.get("cfg")
        sampler_name = meta.get("sampler_name")
        scheduler = meta.get("scheduler")

        output_dir = folder_paths.get_output_directory() if folder_paths else "output"
        if folder_paths is not None:
            full_prefix, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, output_dir, images.shape[2], images.shape[1]
            )
        else:
            full_prefix, filename, counter, subfolder = (filename_prefix, filename_prefix, 0, "")

        results = []
        for frame in images:
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
                steps=steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                cfg=cfg,
                seed=seed,
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
