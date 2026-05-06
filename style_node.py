"""GrimmRibbity Style — visual style picker that applies LoRAs and conditioning.

Sibling to PromptLibrary (the STRING-output gallery): same library, same modal,
same gallery widget, but this node ingests MODEL/CLIP/CONDITIONING, applies the
entry's stored LoRA stack to MODEL+CLIP, encodes the prompt (plus any LoRA
trigger words) with the patched CLIP, and concats the result onto the input
conditioning. Output: patched MODEL/CLIP, the merged positive + negative
CONDITIONING, plus the final prompt text as a STRING for downstream chains.

The Edit Prompt modal grows a "+ Add LoRA" section that writes the loras list
back to the entry — this node reads that list at run time so workflows pick up
edits without rewiring.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

import comfy.sd
import comfy.utils
import folder_paths

# We deliberately import the storage helpers from the package's __init__ so the
# style node and the gallery node share one source of truth (lock, schema,
# load/save). The package is already loaded by the time INPUT_TYPES runs.
from . import _load, _lock


_log = logging.getLogger(__name__)
_NONE_LORA = "(none)"


def _resolve_lora_path(name: str) -> str | None:
    if not name or name == _NONE_LORA:
        return None
    try:
        path = folder_paths.get_full_path("loras", name)
    except Exception:
        return None
    return path if path and os.path.isfile(path) else None


# Keep the most recently used LoRA tensors in memory so a workflow that
# re-runs with the same stack doesn't re-read every safetensors file from
# disk on every queue. Cap at 4 — enough for a typical style + character
# combo, small enough that VRAM/RAM doesn't bloat from idle nodes.
_LORA_CACHE: list[tuple[str, dict]] = []
_LORA_CACHE_MAX = 4
_LORA_CACHE_LOCK = threading.Lock()


def _load_lora_cached(path: str) -> dict:
    with _LORA_CACHE_LOCK:
        for cached_path, sd in _LORA_CACHE:
            if cached_path == path:
                return sd
    # Load outside the lock — file IO can be hundreds of ms and we don't want
    # to serialize concurrent runs that happen to want different LoRAs.
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    with _LORA_CACHE_LOCK:
        # A racing caller may have populated this entry while we were loading;
        # if so use theirs and let our copy be GC'd. Cheap dedupe.
        for cached_path, cached_sd in _LORA_CACHE:
            if cached_path == path:
                return cached_sd
        _LORA_CACHE.append((path, sd))
        while len(_LORA_CACHE) > _LORA_CACHE_MAX:
            _LORA_CACHE.pop(0)
    return sd


def _encode_prompt(clip, text: str):
    """Encode a single prompt into CONDITIONING using whatever encode API the
    CLIP exposes. encode_from_tokens_scheduled is the modern path (SDXL +
    scheduled prompts); fall back to encode_from_tokens for older CLIP wrappers."""
    tokens = clip.tokenize(text)
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _concat_conditioning(base, addition):
    """Token-axis concat — mirrors comfy_extras ConditioningConcat. base may be
    None / empty when the node is used standalone, in which case we just emit
    addition. The first entry of `addition` is applied across every entry in
    `base` (matching core behaviour for a single new prompt)."""
    if not base:
        return addition
    if not addition:
        return base
    if len(addition) > 1:
        _log.warning(
            "PromptLibraryStyle: addition has %d conditioning entries; only "
            "the first will be concatenated (matches core ConditioningConcat).",
            len(addition))
    cond_from = addition[0][0]
    out = []
    for entry in base:
        t1 = entry[0]
        try:
            merged = torch.cat((t1, cond_from), dim=1)
        except Exception:
            # Shape mismatch (mixed model classes, etc.) — keep the base
            # entry untouched rather than crashing the workflow.
            merged = t1
        out.append([merged, entry[1].copy()])
    return out


def _trim_comma_ws(s: str) -> str:
    """Strip trailing whitespace and commas (in any combination) — avoids
    double-comma artefacts when joining text + triggers."""
    return (s or "").strip().rstrip(",").rstrip()


def _format_prompt(text: str, loras: list[dict], extra: str = "") -> str:
    """Combine entry text + enabled trigger words + per-call extra_text into
    a single comma-separated prompt string. Disabled rows and blank-trigger
    rows are skipped."""
    parts: list[str] = []
    if text := _trim_comma_ws(text):
        parts.append(text)
    for l in loras:
        if not l.get("enabled", True):
            continue
        t = _trim_comma_ws(l.get("triggers") or "")
        if t:
            parts.append(t)
    if extra := _trim_comma_ws(extra):
        parts.append(extra)
    return ", ".join(parts)


def _apply_loras(model, clip, loras: list[dict], strength_scale: float
                  ) -> tuple[object, object, list[str]]:
    """Patch model + clip with each enabled LoRA. Returns the patched objects
    plus a list of short status strings for the summary log line."""
    status: list[str] = []
    for l in loras:
        name = l.get("name", "")
        if not l.get("enabled", True):
            status.append(f"-{name}(off)")
            continue
        sm = float(l.get("strength_model", 1.0)) * strength_scale
        sc = float(l.get("strength_clip", l.get("strength_model", 1.0))) * strength_scale
        if sm == 0 and sc == 0:
            status.append(f"-{name}(0)")
            continue
        lora_path = _resolve_lora_path(name)
        if not lora_path:
            status.append(f"!{name}(missing)")
            continue
        try:
            lora_sd = _load_lora_cached(lora_path)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, sm, sc)
            status.append(f"+{name}@{sm:g}")
        except Exception as e:
            _log.warning("PromptLibraryStyle: failed to apply %r: %s", name, e)
            status.append(f"!{name}({type(e).__name__})")
    return model, clip, status


class PromptLibraryStyle:
    """Pick a style entry from the gallery, apply its LoRA stack to MODEL/CLIP,
    encode the prompt (positive + negative) on the patched CLIP, and concat
    the result onto the input conditioning. One node replaces a LoraLoader
    chain plus two CLIPTextEncode nodes for the typical style preset."""

    DESCRIPTION = (
        "Visual style picker that ALSO applies the entry's stored LoRA stack. "
        "Each library entry can hold up to 10 LoRAs (added in the Edit Prompt "
        "modal). This node patches MODEL+CLIP with each one, encodes the entry's "
        "positive prompt (with trigger words appended) AND its stored negative "
        "prompt with the patched CLIP, and concats both onto the optional "
        "positive/negative CONDITIONING inputs."
    )

    SEARCH_ALIASES = [
        "style", "lora", "load lora", "apply style", "library style",
        "prompt library style", "grimmribbity style",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Diffusion model to patch with the entry's LoRA stack."}),
                "clip": ("CLIP", {
                    "tooltip": "CLIP encoder. LoRAs patch this too, and both prompts are "
                               "encoded with the patched copy."}),
                "prompt_id": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Entry id selected by the gallery widget. Driven by clicks; "
                               "you don't normally type here."}),
            },
            "optional": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Existing positive conditioning to concat onto. Leave unwired "
                               "to emit the new conditioning by itself."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "Existing negative conditioning to concat onto. The entry's "
                               "stored negative (if any) is encoded and concatted here."}),
                "extra_text": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Free-form text appended after the entry's prompt + triggers "
                               "before encoding. Use for per-call tweaks without editing the entry."}),
                "strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplier applied to every LoRA's strength_model + strength_clip "
                               "for this run. 0.0 disables all LoRAs without editing the entry."}),
                "bypass": ("BOOLEAN", {"default": False,
                    "tooltip": "When True, skip every LoRA + prompt-encode step and pass MODEL, "
                               "CLIP, and conditioning through unchanged. Use to A/B-compare with "
                               "vs. without the style applied."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "clip", "positive", "negative", "prompt")
    OUTPUT_TOOLTIPS = (
        "MODEL with every enabled LoRA from the entry applied at strength_model.",
        "CLIP with every enabled LoRA applied at strength_clip — used for the "
        "prompt encoding done inside this node.",
        "Input positive conditioning with the entry's encoded prompt concatenated. "
        "If positive was unwired, this is just the new encoded prompt.",
        "Input negative conditioning with the entry's encoded negative concatenated. "
        "If neither was set, this is a trivial empty-prompt encoding.",
        "The final positive prompt string (entry text + extra_text + triggers). "
        "For wiring into separate text-only chains or for debugging.",
    )
    FUNCTION = "apply"
    CATEGORY = "GrimmRibbity/Library"

    @classmethod
    def IS_CHANGED(cls, model, clip, prompt_id, positive=None, negative=None,
                    extra_text="", strength_scale=1.0, bypass=False):
        if bypass:
            return "bypass"
        with _lock:
            entry = next((i for i in _load() if i.get("id") == (prompt_id or "").strip()), None)
        if entry is None:
            return f"miss:{prompt_id}"
        loras = entry.get("loras") or []
        sig = "|".join(
            f"{l.get('name','')}:{l.get('strength_model',1.0):g}:{l.get('strength_clip',1.0):g}:"
            f"{int(bool(l.get('enabled', True)))}:{(l.get('triggers') or '').strip()}"
            for l in loras
        )
        return (f"{entry.get('text','')}::{entry.get('negative','')}::{extra_text}::"
                f"{strength_scale:g}::{sig}")

    def apply(self, model, clip, prompt_id, positive=None, negative=None,
                extra_text="", strength_scale=1.0, bypass=False):
        if bypass:
            # Empty conditioning would crash the sampler — encode "" so both
            # sockets always carry a valid CONDITIONING even when unwired.
            pos_out = positive if positive else _encode_prompt(clip, "")
            neg_out = negative if negative else _encode_prompt(clip, "")
            return (model, clip, pos_out, neg_out, "")

        pid = (prompt_id or "").strip()
        with _lock:
            entry = next((i for i in _load() if i.get("id") == pid), None)
        if entry is None:
            _log.warning("PromptLibraryStyle: no entry with id=%r; passing through", pid)
            pos_out = positive if positive else _encode_prompt(clip, "")
            neg_out = negative if negative else _encode_prompt(clip, "")
            return (model, clip, pos_out, neg_out, "")

        loras = list(entry.get("loras") or [])
        patched_model, patched_clip, lora_status = _apply_loras(
            model, clip, loras, strength_scale)
        full_text = _format_prompt(entry.get("text", ""), loras, extra_text)
        new_pos = _encode_prompt(patched_clip, full_text)
        merged_pos = _concat_conditioning(positive, new_pos)
        # Negative is encoded only when the entry actually carries one — the
        # input negative passes through untouched otherwise. This matches
        # how the Library node treats the negative STRING.
        entry_neg = (entry.get("negative") or "").strip()
        if entry_neg:
            new_neg = _encode_prompt(patched_clip, entry_neg)
            merged_neg = _concat_conditioning(negative, new_neg)
        elif negative:
            merged_neg = negative
        else:
            merged_neg = _encode_prompt(patched_clip, "")

        if lora_status:
            _log.info("PromptLibraryStyle: applied %d LoRA(s) for %r: %s",
                       len(loras), entry.get("name", pid), " ".join(lora_status))
        return (patched_model, patched_clip, merged_pos, merged_neg, full_text)
