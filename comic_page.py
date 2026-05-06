"""GrimmRibbity Comic Page (Regional) node.

Wraps Inspire Pack's RegionalConditioningColorMask + combine-conditionings
pattern into a single node that takes a color-mask layout image and N
per-panel prompts, and emits a single CONDITIONING ready for the sampler.

Pair upstream with `GrimmRibbity — Character Anchor` (or any IPAdapter setup
that produces a MODEL with a global character lock) and this gives you the
per-panel scene/action conditioning for a single-gen multi-panel comic page.

The 2026 research consensus is that per-panel generation with shared
IPAdapter anchor still beats single-gen; this node is the path for users
who explicitly want the single-gen approach (faster, lower per-panel
control). See research/comic_pipeline_2026.md for context.

Hard dependency on ComfyUI-Inspire-Pack — module import fails fast and the
suite registers without the node if Inspire isn't installed.
"""
from __future__ import annotations

# Inspire Pack's folder is `ComfyUI-Inspire-Pack` (hyphens), so Python's
# normal `from custom_nodes...import` won't resolve it. Pull the registered
# class out of ComfyUI's global node registry instead.
#
# Look up at runtime, not at module load: ComfyUI's node-registration order
# isn't deterministic across custom packs, and PromptLibrary often loads
# before Inspire Pack does. If we look up Inspire's class at import time,
# `nodes.NODE_CLASS_MAPPINGS` is empty for Inspire's keys and we silently
# disable Comic Page. Deferring to build() guarantees the dict is fully
# populated by the time a workflow actually runs the node.
import nodes as _comfy_nodes  # type: ignore


def _lookup_regional_cond_color_mask():
    cls = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("RegionalConditioningColorMask")
    if cls is None:
        raise RuntimeError(
            "GrimmRibbity Comic Page requires ComfyUI-Inspire-Pack to be installed and "
            "loaded (it provides RegionalConditioningColorMask). If Inspire Pack IS "
            "installed, restart ComfyUI — node-registration order matters."
        )
    return cls

# Default 2x2 panel colour scheme — rendering this 4-colour mask in any
# image editor (red top-left, green top-right, blue bottom-left, yellow
# bottom-right) is the simplest authoring path.
_DEFAULT_COLORS = ("#FF0000", "#00FF00", "#0000FF", "#FFFF00")
_PANEL_COUNT = 6  # widgets exposed; user can leave any unused (blank prompt)


def _empty_conditioning():
    """Empty CONDITIONING — a list of zero entries. Used as the fallback
    when no panels have prompts (downstream sampler still needs a value)."""
    return []


class GrimmRibbityComicPage:
    """Single-node region-aware conditioning for a multi-panel comic page.

    For each panel: encode its prompt with the supplied CLIP and constrain
    the resulting conditioning to the panel's color region in the layout
    mask. Combine all panel conditionings into one output.
    """

    DESCRIPTION = (
        "Build a single CONDITIONING from a color-coded panel-layout mask "
        "and per-panel prompts. Each prompt is constrained to its region. "
        "Wire CLIP in, panel_layout in (red/green/blue/yellow color mask), "
        "type per-panel prompts, and the output goes into your sampler's "
        "positive input. Empty prompts are skipped — leaving panel_5/6 blank "
        "gives you a 4-panel page; using all 6 gives a denser layout. "
        "Pair with GrimmRibbity Character Anchor for character lock across "
        "panels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP encoder. Wire from your checkpoint loader."}),
                "panel_layout": ("IMAGE", {"tooltip": "Color-coded mask image. Each panel is a "
                                                       "different solid color matching the panel_N_color "
                                                       "fields below. Same dimensions as your final canvas."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "How strongly each panel's prompt binds to its region. 1.0 is normal; "
                               "higher values pull harder. Lower (0.5–0.7) lets panels bleed into each other."}),
                "set_cond_area": (["default", "mask bounds"], {"default": "mask bounds",
                    "tooltip": "'mask bounds' restricts conditioning to the panel's bounding box "
                               "(cleaner separation). 'default' applies to the whole canvas with "
                               "masked attention."}),
            },
            "optional": {},
        }
        for i in range(1, _PANEL_COUNT + 1):
            default_color = _DEFAULT_COLORS[(i - 1) % len(_DEFAULT_COLORS)]
            spec["required"][f"panel_{i}_prompt"] = ("STRING", {
                "default": "", "multiline": True,
                "placeholder": f"panel {i} prompt — leave blank to skip this panel",
                "tooltip": f"Positive prompt for panel {i}. Leave blank to skip "
                           "(panel won't contribute to the conditioning)."})
            spec["required"][f"panel_{i}_color"] = ("STRING", {
                "default": default_color, "multiline": False,
                "tooltip": f"Hex color (#RRGGBB) of panel {i} in the layout mask. "
                           f"Default {default_color}."})
        return spec

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    OUTPUT_TOOLTIPS = ("Combined regional conditioning for all panels with non-empty prompts. "
                        "Wire into your sampler's positive input.",)
    FUNCTION = "build"
    CATEGORY = "GrimmRibbity/Comic"

    def build(self, clip, panel_layout, strength, set_cond_area, **panels):
        regional = _lookup_regional_cond_color_mask()()
        combined: list = []
        active = 0
        for i in range(1, _PANEL_COUNT + 1):
            prompt = (panels.get(f"panel_{i}_prompt") or "").strip()
            color = (panels.get(f"panel_{i}_color") or "").strip()
            if not prompt or not color:
                continue
            # RegionalConditioningColorMask.doit returns (CONDITIONING, MASK) —
            # we only need the conditioning; mask is for downstream use the
            # caller can compute themselves if they need it.
            cond, _mask = regional.doit(
                clip=clip, color_mask=panel_layout, mask_color=color,
                strength=strength, set_cond_area=set_cond_area, prompt=prompt,
            )
            # CONDITIONING is a list of entries; concatenate to combine regions.
            if isinstance(cond, list):
                combined.extend(cond)
            else:
                combined.append(cond)
            active += 1
        if active == 0:
            print("[ComicPage] no panels had prompts — emitting empty CONDITIONING")
            return (_empty_conditioning(),)
        return (combined,)


NODE_CLASS_MAPPINGS = {"GrimmRibbityComicPage": GrimmRibbityComicPage}
NODE_DISPLAY_NAME_MAPPINGS = {"GrimmRibbityComicPage": "GrimmRibbity — Comic Page (Regional)"}
