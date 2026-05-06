"""GrimmRibbity Comic Page (Regional) node — self-contained.

Takes a color-mask layout image + N per-panel prompts and emits a single
CONDITIONING ready for the sampler. Each panel's prompt is encoded with
CLIP and constrained to its color region in the layout mask.

Pair upstream with `GrimmRibbity — Character Anchor` (or any IPAdapter
setup that produces a MODEL with a global character lock) and this gives
you per-panel scene/action conditioning for a single-gen multi-panel page.

The 2026 research consensus is that per-panel generation with a shared
IPAdapter anchor still beats single-gen; this node is the path for users
who explicitly want the single-gen approach (faster, lower per-panel
control). See research/comic_pipeline_2026.md for context.

NO external custom-pack dependencies. Uses only ComfyUI's built-in
CLIPTextEncode and ConditioningSetMask plus torch — both always present
in any ComfyUI install. Color-to-mask conversion is implemented here
rather than imported from Inspire Pack.
"""
from __future__ import annotations

# CLIPTextEncode and ConditioningSetMask are in ComfyUI's nodes.py core,
# registered before any custom packs load. The deferred-lookup pattern
# isn't needed for these — they're guaranteed present at runtime.
import nodes as _comfy_nodes  # type: ignore

# torch is a ComfyUI core dep but the headless test venv (.testenv) skips
# it for speed. Lazy import inside the helper that needs it so the module
# still loads under tests; the helper raises a clear error if torch is
# genuinely missing at runtime.
try:
    import torch as _torch  # type: ignore
except ImportError:
    _torch = None


# Default 2x2 panel colour scheme — rendering this 4-colour mask in any
# image editor (red top-left, green top-right, blue bottom-left, yellow
# bottom-right) is the simplest authoring path.
_DEFAULT_COLORS = ("#FF0000", "#00FF00", "#0000FF", "#FFFF00")
_PANEL_COUNT = 6  # widgets exposed; user can leave any unused (blank prompt)

# Per-channel tolerance when matching a panel's hex color in the layout
# image. Anti-alias artifacts on panel boundaries can shift a pure red
# pixel by a few units; ~10/255 ≈ 0.039 swallows that without bleeding
# into adjacent panels at this contrast level.
_COLOR_MATCH_TOLERANCE = 0.04


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """Parse '#RRGGBB' or 'RRGGBB' into a (r, g, b) tuple in 0–1 range."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected 6-char hex color, got {hex_color!r}")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def _color_to_mask(image, hex_color: str):
    """Build a binary MASK from an IMAGE tensor by matching a target color.

    image: (B, H, W, 3) float in 0–1 (ComfyUI's IMAGE convention)
    returns: (B, H, W) float in {0.0, 1.0} — mask where image≈hex_color
    """
    if _torch is None:
        raise RuntimeError("torch is required at runtime — ComfyUI provides it.")
    r, g, b = _hex_to_rgb01(hex_color)
    target = _torch.tensor([r, g, b], dtype=image.dtype, device=image.device)
    # L1 distance per pixel across the 3 channels — channels-last tensor.
    diff = (image - target).abs().sum(dim=-1)  # (B, H, W)
    return (diff < _COLOR_MATCH_TOLERANCE * 3).to(dtype=image.dtype)


class GrimmRibbityComicPage:
    """Single-node region-aware conditioning for a multi-panel comic page.

    For each panel: encode its prompt with the supplied CLIP and constrain
    the resulting conditioning to the panel's color region in the layout
    mask. Combine all panel conditionings into one output."""

    DESCRIPTION = (
        "Build a single CONDITIONING from a color-coded panel-layout mask "
        "and per-panel prompts. Each prompt is constrained to its region. "
        "Wire CLIP in, panel_layout in (red/green/blue/yellow color mask), "
        "type per-panel prompts, and the output goes into your sampler's "
        "positive input. Empty prompts are skipped — leaving panel_5/6 blank "
        "gives a 4-panel page; using all 6 gives a denser layout. "
        "Pair with GrimmRibbity Character Anchor for character lock across "
        "panels. No third-party node packs required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP encoder. Wire from your checkpoint loader."}),
                "panel_layout": ("IMAGE", {"tooltip": "Color-coded mask image. Each panel is a "
                                                       "different solid color matching the panel_N_color "
                                                       "fields below. Same dimensions as your final canvas."}),
                "shared_prompt": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "character + style traits prepended to EVERY panel — e.g. "
                                    "'jessica vale, blue_hair, red_eyes, comic style'",
                    "tooltip": "Text prepended to every non-empty panel prompt before encoding. "
                               "This is THE knob for character consistency across panels — type the "
                               "character description here once and every panel inherits it. Pair "
                               "with Character Anchor for a face/style lock on top."}),
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

    def build(self, clip, panel_layout, shared_prompt, strength, set_cond_area, **panels):
        encoder = _comfy_nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        masker = _comfy_nodes.NODE_CLASS_MAPPINGS["ConditioningSetMask"]()

        # `shared_prompt` is prepended to each non-empty panel prompt before
        # encoding. This is the consistency knob — character + style traits
        # typed once and inherited by every panel. Joined with ", " so it
        # reads as a comma-separated tag list (matches booru/SDXL convention).
        shared = (shared_prompt or "").strip()

        combined: list = []
        active = 0
        for i in range(1, _PANEL_COUNT + 1):
            prompt = (panels.get(f"panel_{i}_prompt") or "").strip()
            color = (panels.get(f"panel_{i}_color") or "").strip()
            if not prompt or not color:
                continue
            full_prompt = f"{shared}, {prompt}" if shared else prompt
            mask = _color_to_mask(panel_layout, color)
            (cond,) = encoder.encode(clip, full_prompt)
            (cond,) = masker.append(cond, mask, set_cond_area, strength)
            if isinstance(cond, list):
                combined.extend(cond)
            else:
                combined.append(cond)
            active += 1
        if active == 0:
            print("[ComicPage] no panels had prompts — emitting empty CONDITIONING")
            return ([],)
        return (combined,)


NODE_CLASS_MAPPINGS = {"GrimmRibbityComicPage": GrimmRibbityComicPage}
NODE_DISPLAY_NAME_MAPPINGS = {"GrimmRibbityComicPage": "GrimmRibbity — Comic Page (Regional)"}
