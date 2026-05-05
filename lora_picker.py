"""GrimmRibbity LoRA Picker.

A small companion node that:
  1. Lists every LoRA in models/loras/ as a dropdown.
  2. Composes a `<lora:path:weight>` token for the prompt-text chain
     (parsed downstream by rgthree Power Lora Loader / similar).
  3. Lets you attach trigger words and save the combo to the GrimmRibbity
     library on the same run, so you can rediscover the LoRA via the
     normal gallery later.

Output is STRING — wire it into your prompt-text concatenation chain.
The actual LoRA application happens in a downstream node that parses
the token (rgthree, A1111-style processors, etc.) — this node does not
modify MODEL or CLIP itself. That keeps it composable with whatever
loader the user already has wired up.
"""

from __future__ import annotations

import threading
import urllib.request

import folder_paths


_NONE_LORA = "(none)"


def _list_loras() -> list[str]:
    files = folder_paths.get_filename_list("loras") or []
    return [_NONE_LORA] + sorted(files)


def _build_lora_token(lora_name: str, weight: float, *, use_windows_separators: bool) -> str:
    """Format a single inline LoRA token. By default uses the path Comfy
    stored (typically `/` on Linux, `\\` on Windows). The toggle lets the
    user emit Windows-style backslash paths to match a workflow that was
    authored on Windows."""
    if not lora_name or lora_name == _NONE_LORA:
        return ""
    name = lora_name.replace("/", "\\") if use_windows_separators else lora_name
    return f"<lora:{name}:{weight:g}>"


def _slugify(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in (name or "").lower())
    return out.strip("_")[:64] or "lora"


def _save_to_library(*, name: str, text: str, extra_tags: str, notes: str) -> str:
    """POST the entry to the running ComfyUI's library route. Best-effort —
    catches exceptions so a transient HTTP failure doesn't break the
    sampling pass. Returns the saved entry id, or "" on failure."""
    body, content_type = _build_multipart({
        "name": name,
        "text": text,
        "tags": ", ".join(t.strip() for t in
                           ["lora"] + [s for s in extra_tags.split(",") if s.strip()]
                           if t),
        "rating": "0",
        "notes": notes,
    })
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt_library/upsert",
            data=body, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=5) as r:
            import json
            payload = json.load(r)
            return payload.get("id", "")
    except Exception as e:
        print(f"[LoRA Picker] save_to_library failed: {e}")
        return ""


def _build_multipart(fields: dict) -> tuple[bytes, str]:
    boundary = "----GrimmRibbityLoraPicker"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        parts.append(str(v).encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# Track which (name, weight, triggers) we've already saved this session
# to avoid spamming the library when the workflow re-runs unchanged.
_SAVED_ONCE: set[str] = set()
_SAVED_LOCK = threading.Lock()


class GrimmRibbityLoraPicker:
    """Pick a LoRA from your installed set, attach a weight + trigger words,
    optionally store the combo as a library entry, and emit the result as
    a STRING for downstream prompt-text concatenation. Pairs naturally
    with rgthree's Power Lora Loader (which parses `<lora:name:weight>`
    tokens out of the prompt) — this node just composes the token. Use it
    to encode a known LoRA + its triggers as a reusable library entry."""

    DESCRIPTION = (
        "Pick an installed LoRA + weight + trigger words, emit the inline "
        "<lora:name:weight> token plus triggers as a STRING for prompt-text "
        "chains, and (optionally) save the combo to the library tagged 'lora'. "
        "Downstream nodes (rgthree Power Lora Loader, A1111-style processors) "
        "are responsible for actually applying the LoRA — this node only "
        "composes the token."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_list_loras(), {
                    "tooltip": "Pick from your models/loras/ directory. (none) emits an empty "
                               "token, useful for branching."}),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA strength applied to MODEL. Most LoRAs sit in the 0.6–1.0 "
                               "range. Negative values invert the effect."}),
                "trigger_words": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "tatsumaki, green hair, green eyes, white sclera",
                    "tooltip": "Comma-separated tags that the LoRA was trained to respond to. "
                               "Appended after the lora token in the prompt output."}),
                "use_windows_separators": ("BOOLEAN", {"default": False,
                    "tooltip": "Emit the lora path with backslashes (Windows-style) instead of "
                               "forward slashes. Useful if your downstream loader expects "
                               "Windows paths or if you copied prompts from a Windows-authored "
                               "workflow."}),
                "save_to_library": ("BOOLEAN", {"default": False,
                    "tooltip": "When True, also upsert this combo into the GrimmRibbity library "
                               "as a 'lora'-tagged entry on the next sample. Idempotent within a "
                               "session — the same combo only saves once."}),
                "library_name": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "leave blank to use the LoRA's filename (without extension)",
                    "tooltip": "Custom name for the saved library entry. Empty = derived from "
                               "the LoRA filename."}),
                "library_tags": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "character, anime, sdxl",
                    "tooltip": "Extra comma-separated tags for the saved entry, in addition "
                               "to 'lora'."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "lora_token", "triggers")
    OUTPUT_TOOLTIPS = (
        "Full string: '<lora:name:weight>, trigger words'. Wire into your prompt-text "
        "chain (or directly into rgthree Power Lora Loader's prompt input).",
        "Just the lora token, no triggers. Useful when you want to feed triggers "
        "and the lora into separate parts of your prompt.",
        "Just the triggers, no lora token. Useful when something else handles the "
        "lora wiring and you only want the keywords here.",
    )
    FUNCTION = "pick"
    CATEGORY = "utils"

    def pick(self, lora_name, weight, trigger_words, use_windows_separators,
              save_to_library, library_name, library_tags):
        token = _build_lora_token(lora_name, weight,
                                    use_windows_separators=use_windows_separators)
        triggers = (trigger_words or "").strip()
        if token and triggers:
            full = f"{token}, {triggers}"
        else:
            full = token or triggers

        if save_to_library and lora_name != _NONE_LORA:
            base = (library_name or "").strip()
            if not base:
                # Derive from filename: drop extension + dirs.
                base = lora_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                base = base.rsplit(".", 1)[0]
            cache_key = f"{base}|{token}|{triggers}"
            with _SAVED_LOCK:
                already = cache_key in _SAVED_ONCE
                if not already:
                    _SAVED_ONCE.add(cache_key)
            if not already:
                notes = (f"LoRA: {lora_name}\nWeight: {weight:g}\n"
                         f"Auto-saved by GrimmRibbity LoRA Picker.")
                saved_id = _save_to_library(name=base, text=full,
                                              extra_tags=library_tags, notes=notes)
                if saved_id:
                    print(f"[LoRA Picker] saved '{base}' to library as id={saved_id}")
        return (full, token, triggers)
