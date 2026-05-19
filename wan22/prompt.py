"""Wan2.2 prompt builder.

Reuses the PromptLibrary wildcard engine (so __character__ / {a|b|c} refs
expand against the same prompts.json the rest of the suite uses) and tacks on
motion-vocab presets the user can pick from dropdowns — camera move, shot
length, FPS hint, vibe — chosen for Wan2.2's i2v sensibilities (it responds
well to explicit camera direction tokens).

Output is plain STRING for positive + negative; encode them with CLIPTextEncode
downstream so the user keeps full control over the conditioning pipeline.
"""

from __future__ import annotations

import random

# Pull the shared engine from the package root — same source of truth as
# PromptLibraryWildcard etc.
from .. import STORE_PATH, _expand_wildcards, _load, _lock


_INT_MAX = 0xFFFFFFFFFFFFFFFF
_NONE = "(none)"


# Motion vocabulary — keep each list short and unambiguous; first entry is
# the "skip" sentinel and gets dropped from the joined output.
_CAMERA_MOVES = [
    _NONE,
    "static camera",
    "slow push in",
    "slow pull out",
    "slow pan left",
    "slow pan right",
    "slow tilt up",
    "slow tilt down",
    "tracking shot",
    "dolly zoom",
    "orbit shot",
    "handheld shake",
    "crane up",
    "crane down",
]

_SHOT_LENGTHS = [
    _NONE,
    "close-up",
    "medium close-up",
    "medium shot",
    "medium wide shot",
    "wide shot",
    "extreme wide shot",
    "over the shoulder",
    "low angle",
    "high angle",
    "dutch angle",
]

_FPS_HINTS = [
    _NONE,
    "24 fps cinematic",
    "30 fps smooth",
    "60 fps high frame rate",
    "slow motion",
    "time lapse",
]

_VIBES = [
    _NONE,
    "cinematic lighting",
    "soft natural light",
    "golden hour",
    "blue hour",
    "neon noir",
    "overcast diffuse light",
    "dramatic rim light",
    "volumetric fog",
]

_DEFAULT_NEGATIVE = (
    "static, frozen, low quality, blurry, distorted, watermark, text, "
    "subtitle, jpeg artifacts, oversaturated, deformed, malformed, "
    "extra limbs, extra fingers, bad anatomy"
)


def _join_motion(*parts: str) -> str:
    keep = [p.strip() for p in parts if p and p != _NONE]
    return ", ".join(keep)


def _comma_join(a: str, b: str) -> str:
    a = (a or "").strip().rstrip(",").rstrip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a}, {b}"


class GrimmRibbityWan22PromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Subject prompt. Supports PromptLibrary "
                               "wildcards: {a|b|c} for alternatives, "
                               "__name__ for library refs."}),
                "negative": ("STRING", {"default": _DEFAULT_NEGATIVE,
                    "multiline": True,
                    "tooltip": "Negative prompt. Default covers the usual "
                               "Wan2.2 i2v failure modes (frozen frames, "
                               "extra limbs). Same wildcard syntax."}),
                "camera_move": (_CAMERA_MOVES, {
                    "tooltip": "Appended verbatim to the positive prompt. "
                               "Wan2.2 i2v reads camera tokens reliably."}),
                "shot_length": (_SHOT_LENGTHS,),
                "fps_hint": (_FPS_HINTS, {
                    "tooltip": "Stylistic FPS / time hint — does not change "
                               "actual frame count."}),
                "vibe": (_VIBES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX,
                    "control_after_generate": True,
                    "tooltip": "Drives wildcard picks. Same seed = same "
                               "expansion across runs."}),
            },
            "optional": {
                "expand_wildcards": ("BOOLEAN", {"default": True,
                    "tooltip": "Off = leave {a|b|c} and __name__ literal."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_TOOLTIPS = (
        "Expanded positive prompt with motion vocab appended.",
        "Expanded negative prompt.",
    )
    FUNCTION = "build"
    CATEGORY = "GrimmRibbity/Wan2.2"
    DESCRIPTION = (
        "Builds Wan2.2 i2v prompts using the shared PromptLibrary wildcard "
        "engine, then appends motion-vocab tokens (camera move, shot length, "
        "FPS hint, vibe). Output is plain STRING — encode with CLIPTextEncode."
    )

    @classmethod
    def IS_CHANGED(cls, positive, negative, camera_move, shot_length,
                   fps_hint, vibe, seed, expand_wildcards=True):
        lib_sig = ""
        if expand_wildcards:
            try:
                lib_sig = str(STORE_PATH.stat().st_mtime
                              if STORE_PATH.exists() else 0.0)
            except OSError:
                lib_sig = ""
        return (f"{seed}|{expand_wildcards}|{lib_sig}|{positive}|{negative}"
                f"|{camera_move}|{shot_length}|{fps_hint}|{vibe}")

    def build(self, positive, negative, camera_move, shot_length, fps_hint,
              vibe, seed, expand_wildcards=True):
        with _lock:
            items = _load() if expand_wildcards else []
        rng_pos = random.Random(seed)
        rng_neg = random.Random(seed ^ 0xA5A5A5A5)

        if expand_wildcards:
            pos_text = _expand_wildcards(positive, items, rng_pos)
            neg_text = _expand_wildcards(negative, items, rng_neg)
        else:
            pos_text = positive
            neg_text = negative

        motion = _join_motion(camera_move, shot_length, fps_hint, vibe)
        pos_text = _comma_join(pos_text, motion)

        return (pos_text, neg_text)
