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
from pathlib import Path

try:
    import folder_paths
except ImportError:  # standalone import for tests
    folder_paths = None

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
HASH_CACHE_PATH = DATA_DIR / "hash_cache.json"


def _build_civitai_filename(base: str, counter: int, frame_idx: int,
                            batch_size: int, append_counter: bool) -> str:
    """Pick the per-frame filename for one frame in a batch.

    append_counter=True (default) matches Comfy's SaveImage scheme:
    ``{base}_{NNNNN}_.png`` so a fresh save resumes from the highest
    existing index in the folder and never overwrites prior work.

    append_counter=False writes ``{base}.png`` for a single-frame batch —
    the user owns the name and an existing file at that path will be
    overwritten. For batches >1, frames are still suffixed with a small
    index (``{base}_{i:02d}.png``); otherwise N frames in one save would
    overwrite each other inside the same call. The caller decides when
    to use this mode; the helper just renders the string.
    """
    if append_counter:
        return f"{base}_{counter:05d}_.png"
    if batch_size <= 1:
        return f"{base}.png"
    return f"{base}_{frame_idx:02d}.png"

# The folders that hold full checkpoints / single-file models. `unet/` is the
# legacy name for what is now `diffusion_models/` on newer ComfyUIs; we list
# both because installs with --extra-model-paths can populate either.
_MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet")
_LORA_FOLDER = "loras"

_NONE_LABEL = "(none)"
_PREFIX_SEP = "::"  # "checkpoints::anima_v4.safetensors"

_cache_lock = threading.Lock()

# Cap on the on-disk hash cache so a user with hundreds of historical
# model swaps doesn't grow this file unbounded. Each entry is ~80 bytes
# (key + sha256 hex), so 4096 entries ≈ 320 KB on disk — covers years
# of normal use while still pruning long-stale entries via LRU eviction
# on overflow. Eviction order is by oldest insertion in the on-disk
# JSON, which is preserved as Python dict insertion order on save.
_HASH_CACHE_MAX_ENTRIES = 4096

# Recursion caps on workflow-trace walking. Prompts that legitimately
# chain text or pipe nodes typically need 2-3 hops; higher caps cover
# unusual third-party node packs without risking a stack-overflow on a
# pathological graph (which Comfy would have already refused to execute).
_TEXT_LINK_MAX_DEPTH = 4
_PIPE_TRACE_MAX_DEPTH = 8


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
    # LRU eviction when the cache exceeds the cap — drop the oldest
    # entries (insertion order). Python dicts preserve insertion order
    # since 3.7, so we don't need a separate ordering structure.
    if len(cache) > _HASH_CACHE_MAX_ENTRIES:
        keep = list(cache.items())[-_HASH_CACHE_MAX_ENTRIES:]
        cache = dict(keep)
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
    """SHA256 a file in 1 MB chunks. First-time hash of a 5-15 GB checkpoint
    takes ~30s on fast SSD — check for interrupt between chunks so Cancel
    stays responsive. The check is best-effort: comfy not importable
    (test/standalone) means the loop is uninterruptible but still bounded
    by file size."""
    try:
        import comfy.model_management as _mm
        check_interrupt = _mm.throw_exception_if_processing_interrupted
    except (ImportError, ModuleNotFoundError, AttributeError):
        check_interrupt = None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if check_interrupt is not None:
                check_interrupt()
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
# rgthree's Lora Loader Stack uses fixed slots `lora_01`/`strength_01` …
# `lora_04`/`strength_04`. Skipped when the value is "None".
_RGTHREE_LORA_STACK_TYPE = "Lora Loader Stack (rgthree)"
_KSAMPLER_TYPES = {
    "KSampler", "KSamplerAdvanced",
    "SamplerCustom", "SamplerCustomAdvanced",
    "KSampler (Efficient)",
    "KSampler SDXL (Eff.)",
    "KSampler Adv. (Efficient)",
    # ComfyUI-Easy-Use samplers — they take a `pipe` instead of model/positive/
    # negative; pipe traces back to an easy* loader that holds the prompts.
    "easy fullkSampler", "easy kSampler", "easy kSamplerCustom",
    "easy kSamplerTiled", "easy kSamplerInpainting",
    "easy kSamplerDownscaleUnet", "easy kSamplerSDTurbo",
    "easy unSampler",
    # Our own samplers — SamplerSDXL routes through `sdxl_tuple`, AnimaSampler
    # uses standard model/positive/negative inputs (with `noise_seed` instead
    # of `seed`, already handled in _SAMPLER_PARAM_KEYS).
    "GrimmRibbitySamplerSDXL",
    "GrimmRibbityAnimaSampler",
}
# The set above that uses a `pipe` input rather than direct model/pos/neg links.
_EASY_PIPE_SAMPLER_TYPES = {
    "easy fullkSampler", "easy kSampler", "easy kSamplerCustom",
    "easy kSamplerTiled", "easy kSamplerInpainting",
    "easy kSamplerDownscaleUnet", "easy kSamplerSDTurbo",
    "easy unSampler",
}
# Easy-Use loaders bundle ckpt_name, lora_name, positive, negative as direct
# inputs and emit a `pipe`. The samplers above read everything via that pipe.
_EASY_PIPE_LOADER_TYPES = {
    "easy fullLoader", "easy a1111Loader", "easy comfyLoader",
    "easy fluxLoader", "easy hunyuanDiTLoader", "easy pixArtLoader",
    "easy cascadeLoader", "easy kolorsLoader", "easy mochiLoader",
}
_TEXT_ENCODE_TYPES = {
    "CLIPTextEncode", "CLIPTextEncodeSDXL",
    "BNK_CLIPTextEncodeAdvanced",
    "ImpactWildcardEncode",  # Impact Pack — has populated_text + wildcard_text
}
# Passthrough nodes whose `model` input chains back toward the loader. We walk
# straight through them when looking for the checkpoint, ignoring the node's
# own params. Helpful for things like ModelSamplingDiscrete or wildcard
# encoders that pass MODEL through unchanged.
_MODEL_PASSTHROUGH_TYPES = {
    "ImpactWildcardEncode",
    "ModelSamplingDiscrete", "ModelSamplingSD3", "ModelSamplingFlux",
    "ModelSamplingStableCascade", "ModelSamplingAuraFlow",
    "FreeU", "FreeU_V2", "PerturbedAttentionGuidance",
    "RescaleCFG", "PerpNeg",
}
# Tuple-pack nodes that bundle base_model/base_positive/base_negative into a
# single SDXL_TUPLE wire. SDXL samplers read everything back through that wire
# instead of having direct model/pos/neg inputs.
_SDXL_TUPLE_PACK_TYPES = {
    "Pack SDXL Tuple",         # Efficiency Nodes
    "GrimmRibbityPackSDXLTuple",  # this repo's pack node
}


def _link_source(value):
    """Extract the source node id from a connection link [node_id, output_idx]."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def _resolve_sampler_links(prompt: dict, sampler_node: dict) -> tuple:
    """Return (model_link, positive_link, negative_link) for a sampler, hopping
    through `Pack SDXL Tuple` when an Efficiency-Nodes SDXL sampler routes
    everything through `sdxl_tuple`."""
    inputs = sampler_node.get("inputs") or {}
    model_link = inputs.get("model")
    pos_link = inputs.get("positive")
    neg_link = inputs.get("negative")
    if model_link is not None or pos_link is not None or neg_link is not None:
        return model_link, pos_link, neg_link
    tuple_src = _link_source(inputs.get("sdxl_tuple"))
    if tuple_src and tuple_src in prompt:
        pack = prompt[tuple_src]
        if (pack or {}).get("class_type") in _SDXL_TUPLE_PACK_TYPES:
            p_in = pack.get("inputs") or {}
            return (p_in.get("base_model"),
                    p_in.get("base_positive"),
                    p_in.get("base_negative"))
    return model_link, pos_link, neg_link


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

        if ctype == _RGTHREE_LORA_STACK_TYPE:
            # Fixed 4-slot stack: lora_01..lora_04 + strength_01..strength_04.
            # Apply order is 01 -> 04, so collect in reverse for the outer
            # `reversed()` to flip back into application order.
            slot_loras: list[tuple[str, float]] = []
            for i in range(1, 5):
                lname = inputs.get(f"lora_0{i}") or inputs.get(f"lora_{i:02d}")
                if not isinstance(lname, str) or lname in (None, "None", ""):
                    continue
                strength_raw = inputs.get(f"strength_0{i}", inputs.get(f"strength_{i:02d}", 1.0))
                try:
                    strength = float(strength_raw)
                except (TypeError, ValueError):
                    strength = 1.0
                if strength == 0:
                    continue
                slot_loras.append((lname, strength))
            for entry in reversed(slot_loras):
                loras.append(entry)
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

        if ctype in _SDXL_TUPLE_PACK_TYPES:
            current = _link_source(inputs.get("base_model"))
            continue

        if ctype in _MODEL_PASSTHROUGH_TYPES:
            current = _link_source(inputs.get("model"))
            continue

        # Unknown node — give the `model` input one last try (covers most
        # third-party passthroughs we haven't catalogued).
        current = _link_source(inputs.get("model"))
    return None, list(reversed(loras))


def _resolve_text_link(prompt: dict, link_value, depth: int = 0) -> str:
    """Follow a CLIPTextEncode (or wildcard / Searge text node) link and
    return the prompt text. Tolerates a few levels of indirection — primitive
    Text nodes feeding encoders, Searge prompt nodes, etc.
    """
    if depth > _TEXT_LINK_MAX_DEPTH:
        return ""
    src = _link_source(link_value)
    if not src or src not in prompt:
        return ""
    node = prompt[src] or {}
    ctype = node.get("class_type") or ""
    inputs = node.get("inputs") or {}

    # Impact's wildcard encoder fills `populated_text` with the resolved
    # output (after wildcard substitution); fall back to the template if
    # populated_text is empty (workflow saved before first run).
    if ctype == "ImpactWildcardEncode":
        for key in ("populated_text", "wildcard_text"):
            v = inputs.get(key)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, list):
                resolved = _resolve_text_link(prompt, v, depth + 1)
                if resolved:
                    return resolved

    # pythongosssss/comfyui-custom-scripts StringFunction concatenates or
    # replaces — preserve the result by joining text_a/b/c on the action.
    if ctype == "StringFunction|pysssss":
        action = inputs.get("action", "append")
        parts = []
        for key in ("text_a", "text_b", "text_c"):
            v = inputs.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v)
            elif isinstance(v, list):
                resolved = _resolve_text_link(prompt, v, depth + 1)
                if resolved:
                    parts.append(resolved)
        if not parts:
            return ""
        if action == "replace" and len(parts) >= 1:
            # Best-effort: emit the source text; the substitution semantics
            # need runtime evaluation to reproduce exactly.
            return parts[0]
        return ", ".join(parts)

    if ctype in _TEXT_ENCODE_TYPES:
        for key in ("text", "text_g", "text_l"):
            v = inputs.get(key)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, list):
                resolved = _resolve_text_link(prompt, v, depth + 1)
                if resolved:
                    return resolved

    # Generic string-output nodes — Searge uses `prompt`, others use `text`/
    # `string`/`value`. Try all.
    for key in ("text", "prompt", "string", "value", "wildcard_text"):
        v = inputs.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list):
            resolved = _resolve_text_link(prompt, v, depth + 1)
            if resolved:
                return resolved
    return ""


_SAMPLER_PARAM_KEYS = ("seed", "steps", "cfg", "sampler_name", "scheduler",
                        "noise_seed")


def _trace_pipe_to_loader(prompt: dict, pipe_link, depth: int = 0):
    """Follow a `pipe` link back to the originating Easy-Use loader. Easy-Use
    samplers may chain through intermediate pipe-routing nodes (e.g. branch,
    edit), all of which expose a `pipe` input pointing upstream."""
    if depth > _PIPE_TRACE_MAX_DEPTH:
        return None
    src = _link_source(pipe_link)
    if not src or src not in prompt:
        return None
    node = prompt[src] or {}
    if (node.get("class_type") or "") in _EASY_PIPE_LOADER_TYPES:
        return node
    inputs = node.get("inputs") or {}
    return _trace_pipe_to_loader(prompt, inputs.get("pipe"), depth + 1)


def _populate_from_easy_loader(out: dict, prompt: dict, loader: dict) -> None:
    inputs = loader.get("inputs") or {}
    ckpt = inputs.get("ckpt_name")
    if isinstance(ckpt, str) and ckpt and ckpt != "None":
        out["model_label"] = f"checkpoints{_PREFIX_SEP}{ckpt}"

    loras: list[tuple[str, float]] = []
    lname = inputs.get("lora_name")
    if isinstance(lname, str) and lname not in (None, "None", ""):
        try:
            strength = float(inputs.get("lora_model_strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        if strength != 0:
            loras.append((lname, strength))

    # `optional_lora_stack` may chain through an `easy loraStack` or rgthree
    # `Lora Loader Stack`; walk the same model-chain logic against that link
    # to collect more.
    stack_link = inputs.get("optional_lora_stack")
    stack_src = _link_source(stack_link)
    if stack_src:
        # Reuse _walk_model_chain: it'll handle LoraLoader / LoraLoaderStack /
        # PowerLoraLoader uniformly. The chain stops as soon as it can't find
        # a `model` link, which is fine for STACK-typed nodes that don't
        # forward a model.
        _, stack_loras = _walk_model_chain(prompt, stack_src)
        loras.extend(stack_loras)

    if loras:
        out["loras"] = loras

    pos = inputs.get("positive")
    if isinstance(pos, str) and pos.strip():
        out["positive"] = pos
    elif isinstance(pos, list):
        resolved = _resolve_text_link(prompt, pos)
        if resolved:
            out["positive"] = resolved
    neg = inputs.get("negative")
    if isinstance(neg, str) and neg.strip():
        out["negative"] = neg
    elif isinstance(neg, list):
        resolved = _resolve_text_link(prompt, neg)
        if resolved:
            out["negative"] = resolved


def _resolve_literal(prompt: dict, value, *, candidate_keys: tuple[str, ...] = (), max_hops: int = 4):
    """If `value` is a [src_id, idx] connection, walk back through up to
    `max_hops` source nodes looking for a literal in any of `candidate_keys`
    (or the source's own inputs if no candidates given). Returns the literal
    or None. Used for sampler params like seed/cfg/steps that are commonly
    wired from a primitive Seed / Easy-Use Seed / rgthree Seed node instead
    of typed in directly.
    """
    visited: set[str] = set()
    cur = value
    for _ in range(max_hops):
        if not isinstance(cur, list) or len(cur) != 2:
            break
        src_id = str(cur[0])
        if src_id in visited:
            return None
        visited.add(src_id)
        src = prompt.get(src_id)
        if not isinstance(src, dict):
            return None
        src_in = src.get("inputs") or {}
        # Try the candidate keys first (e.g., "seed"/"noise_seed"/"value" for
        # seed nodes), then any non-link input as a last resort.
        keys = list(candidate_keys) or list(src_in.keys())
        for k in keys:
            if k not in src_in:
                continue
            v = src_in[k]
            if not isinstance(v, list):
                return v
        # Source had no literal but might itself be a passthrough — pick the
        # first wired input matching candidate_keys and recurse one hop.
        next_step = None
        for k in (candidate_keys or src_in.keys()):
            v = src_in.get(k)
            if isinstance(v, list) and len(v) == 2:
                next_step = v
                break
        if next_step is None:
            return None
        cur = next_step
    return None


def _coerce_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v == int(v) else None
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


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
        ctype = sampler.get("class_type")
        s_in = sampler.get("inputs") or {}
        # Per-key fallback chains for following wired connections back to a
        # literal. Seeds in particular are frequently driven by a separate
        # Seed/Primitive node; before we were silently dropping them.
        _RESOLVE_KEYS: dict[str, tuple[str, ...]] = {
            "seed": ("seed", "value", "int", "noise_seed"),
            "noise_seed": ("noise_seed", "seed", "value", "int"),
            "steps": ("steps", "value", "int"),
            "cfg": ("cfg", "value", "float", "number"),
            "sampler_name": ("sampler_name", "value", "string"),
            "scheduler": ("scheduler", "value", "string"),
        }
        for key in _SAMPLER_PARAM_KEYS:
            if key not in s_in:
                continue
            raw = s_in[key]
            if isinstance(raw, list):
                resolved = _resolve_literal(prompt, raw,
                                             candidate_keys=_RESOLVE_KEYS.get(key, (key,)))
                if resolved is not None and not isinstance(resolved, list):
                    out[key] = resolved
            else:
                out[key] = raw

        if ctype in _EASY_PIPE_SAMPLER_TYPES:
            loader = _trace_pipe_to_loader(prompt, s_in.get("pipe"))
            if loader is not None:
                _populate_from_easy_loader(out, prompt, loader)
        else:
            model_link, pos_link, neg_link = _resolve_sampler_links(prompt, sampler)
            model_label, loras = _walk_model_chain(prompt, _link_source(model_link))
            if model_label:
                out["model_label"] = model_label
            if loras:
                out["loras"] = loras
            pos = _resolve_text_link(prompt, pos_link)
            neg = _resolve_text_link(prompt, neg_link)
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

    DESCRIPTION = (
        "SaveImage replacement that writes A1111/Civitai-compatible PNG metadata. "
        "Auto-detects model, LoRAs, positive/negative, seed, sampler, and scheduler "
        "from the workflow trace. Override any field if auto-detection picks the "
        "wrong sampler in a multi-KSampler workflow. Drops the file into "
        "ComfyUI/output by default — use 'output_path' to redirect."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The IMAGE batch to save. Each frame becomes one PNG."}),
                "filename_prefix": ("STRING", {"default": "GrimmRibbity",
                    "tooltip": "Prefix for the saved file (counter and .png are appended). "
                               "Supports ComfyUI's date/time substitutions like %date:yyyy-MM-dd%."}),
                "append_counter": ("BOOLEAN", {"default": True,
                    "tooltip": "When True (default), append Comfy's zero-padded counter "
                               "to the filename (e.g. 'GrimmRibbity_00007_.png') so each "
                               "save resumes from the highest existing index and never "
                               "overwrites prior work. When False, write the filename "
                               "exactly as given (e.g. 'GrimmRibbity.png') — an existing "
                               "file at that path WILL be overwritten. For batch saves "
                               "of N>1 frames, a small index suffix (_00, _01, …) is "
                               "still appended so the frames within one save don't "
                               "clobber each other."}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "multiline": False,
                                            "placeholder": "leave blank for ComfyUI/output, or absolute / ~ / relative path",
                    "tooltip": "Custom save directory. Empty = ComfyUI/output (default). "
                               "Accepts absolute paths, ~-relative, or output-relative. "
                               "Created if missing; counter resumes from highest existing image."}),
                "seed_override": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "Pin the seed in saved metadata. -1 = auto-detect from the workflow "
                               "(default). Use this when auto-detection picks the wrong KSampler in "
                               "a multi-sampler workflow."}),
                "model_override": ([_AUTO_LABEL] + list_all_model_choices()[1:], {
                    "tooltip": "Pin the saved model name. (auto) follows the workflow trace."}),
                "positive_override": ("STRING", {"default": "", "multiline": True,
                                                  "placeholder": "leave blank to auto-detect",
                    "tooltip": "Pin the positive prompt in metadata. Useful when the workflow's "
                               "actual prompt is stitched together from multiple sources and you "
                               "want a clean Civitai page."}),
                "negative_override": ("STRING", {"default": "", "multiline": True,
                                                  "placeholder": "leave blank to auto-detect",
                    "tooltip": "Pin the negative prompt in metadata."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    OUTPUT_TOOLTIPS = (
        "The actually-used seed (override if set, else auto-detected, else 0). "
        "Wire into a logger, filename builder, or another node's seed input.",
    )
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "GrimmRibbity/Output"

    def save(self, images, filename_prefix,
             append_counter=True,
             output_path="",
             seed_override=-1,
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
        # Seed: explicit override wins (anything >= 0); -1 falls back to the
        # auto-detected value from the workflow trace.
        seed = seed_override if (isinstance(seed_override, int) and seed_override >= 0) else meta.get("seed")
        steps = meta.get("steps")
        cfg = meta.get("cfg")
        sampler_name = meta.get("sampler_name")
        scheduler = meta.get("scheduler")

        default_output = folder_paths.get_output_directory() if folder_paths else "output"
        custom = (output_path or "").strip()
        if custom:
            resolved = os.path.expanduser(custom)
            if not os.path.isabs(resolved):
                resolved = os.path.join(default_output, resolved)
            os.makedirs(resolved, exist_ok=True)
            output_dir = resolved
        else:
            output_dir = default_output
        if folder_paths is not None and not custom:
            full_prefix, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, output_dir, images.shape[2], images.shape[1]
            )
        else:
            # Custom output_path bypasses Comfy's subfolder/counter logic so the
            # files land exactly in the directory the user specified. Match the
            # standard SaveImage counter scheme by scanning for existing files.
            base = os.path.basename(filename_prefix) or "image"
            existing = []
            try:
                for f in os.listdir(output_dir):
                    if f.startswith(base + "_") and f.endswith(".png"):
                        try:
                            existing.append(int(f[len(base) + 1:].split("_")[0]))
                        except ValueError:
                            pass
            except FileNotFoundError:
                pass
            counter = (max(existing) + 1) if existing else 0
            full_prefix, filename, subfolder = output_dir, base, ""

        # Parts of the metadata that DON'T vary per frame in a batch are lifted
        # out of the loop. Frames in a Comfy batch always share H/W (the
        # tensor is [B, H, W, C]), so build_a1111_parameters produces the
        # same string for every frame — building it N times is wasted work.
        # Same for the PROMPT / EXTRA_PNGINFO JSON encodes.
        # Width/height come from images.shape since all frames share them;
        # we sample frame[0]'s dims as the canonical source.
        if len(images) > 0:
            sample = images[0]
            sample_h = int(sample.shape[-3]) if hasattr(sample, "shape") else 0
            sample_w = int(sample.shape[-2]) if hasattr(sample, "shape") else 0
        else:
            sample_h = sample_w = 0
        params = build_a1111_parameters(
            positive=positive, negative=negative,
            width=sample_w, height=sample_h,
            steps=steps, sampler_name=sampler_name, scheduler=scheduler,
            cfg=cfg, seed=seed,
            model_name=model_name, model_sha256=model_sha,
            loras=loras,
        )
        prompt_text = json.dumps(prompt) if prompt is not None else None
        extra_text: list[tuple[str, str]] = []
        if extra_pnginfo is not None:
            for k, v in extra_pnginfo.items():
                extra_text.append((k, json.dumps(v)))

        try:
            import comfy.model_management as _mm
            check_interrupt = _mm.throw_exception_if_processing_interrupted
        except (ImportError, ModuleNotFoundError, AttributeError):
            check_interrupt = None

        results = []
        batch_size = len(images)
        for frame_idx, frame in enumerate(images):
            # PNG encode + write of a 4K image takes 1-2 s; a 100-frame batch
            # without a cancel-check would ignore the Interrupt button for
            # the whole save. Per-frame check keeps Cancel responsive.
            if check_interrupt is not None:
                check_interrupt()
            arr = frame
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("parameters", params)
            if prompt_text is not None:
                png_info.add_text("prompt", prompt_text)
            for k, v in extra_text:
                png_info.add_text(k, v)

            file_name = _build_civitai_filename(
                filename, counter, frame_idx, batch_size, append_counter)
            counter += 1
            full_path = os.path.join(full_prefix, file_name)
            pil.save(full_path, format="PNG", pnginfo=png_info, compress_level=4)
            results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": "output",
            })

        # OUTPUT_NODE = True puts the previews in the UI; the result tuple
        # under "result" feeds the seed output port for downstream wiring.
        # Fall back to 0 (not -1) when no seed was found — many other nodes
        # declare their seed input with min=0, and emitting -1 here trips
        # ComfyUI's wire-validation ("Value -1 smaller than min of 0").
        return {"ui": {"images": results}, "result": (int(seed) if seed is not None else 0,)}
