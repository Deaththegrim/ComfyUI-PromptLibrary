"""GrimmRibbity Character Anchor node.

Wraps the 3-node IPAdapter chain (UnifiedLoader → IPAdapter Apply → optional
LoadImage) into a single MODEL → MODEL transform so workflows don't have to
spaghetti-wire the same pattern every time.

Hard dependency on ComfyUI_IPAdapter_plus — the wrapping is intentional, we
defer all the heavy lifting (CLIP-vision, weight curves, attention injection)
to the upstream pack. If IPAdapter Plus isn't installed, this module raises
ImportError at load time and __init__.py registers the rest of the suite as
usual without it.
"""
from __future__ import annotations

# Lazy import: __init__.py wraps this file's load in try/except so the suite
# still registers if IPAdapter Plus isn't installed.
from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import (  # type: ignore
    IPAdapterUnifiedLoader,
    IPAdapterSimple,
)


_PRESETS = (
    "PLUS FACE (portraits)",
    "PLUS (high strength)",
    "STANDARD (medium strength)",
    "VIT-G (medium strength)",
    "LIGHT - SD1.5 only (low strength)",
    "FULL FACE - SD1.5 only (portraits stronger)",
)

# IPAdapterSimple's three weight types; lower-level Advanced node has more
# but Simple covers the comic / character-anchor use case cleanly.
_WEIGHT_TYPES = ("standard", "prompt is more important", "style transfer")


class GrimmRibbityCharacterAnchor:
    """Apply an IPAdapter character reference to a MODEL chain in one node.

    Takes a MODEL (typically from your LoRA loader) and an IMAGE (the
    character reference) and emits a MODEL with the IPAdapter conditioning
    applied. Set bypass=True to disable without unwiring."""

    DESCRIPTION = (
        "Pin a character's face/style across panels. Wraps IPAdapter Plus's "
        "UnifiedLoader + Apply pair into a single node so you don't need to "
        "wire 3 nodes per workflow. Wire MODEL in (after your LoRA loader), "
        "wire IMAGE in (the anchor / reference image), and route MODEL out "
        "into your sampler. Toggle bypass=True to disable without rewiring."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MODEL chain. Wire from your LoRA loader's MODEL output."}),
                "preset": (list(_PRESETS), {"default": "PLUS FACE (portraits)",
                    "tooltip": "IPAdapter Plus preset. PLUS FACE for portraits is the right pick "
                               "for character consistency across comic panels. PLUS for general "
                               "appearance lock. STANDARD for lighter influence."}),
                "weight": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "How strongly the reference influences the gen. 0.7 is balanced; "
                               "0.85+ for tight face lock; 0.4-0.5 for subtle style hint."}),
                "weight_type": (list(_WEIGHT_TYPES), {"default": "standard",
                    "tooltip": "'standard' is linear blending. 'prompt is more important' lets the "
                               "text prompt override. 'style transfer' transfers style without identity."}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Step ratio at which IPAdapter starts influencing. 0 = from start. "
                               "Try 0.2 if the reference is dominating composition too much."}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Step ratio at which IPAdapter stops. 1 = until the end. Drop to 0.7 "
                               "to let the final steps render free of the reference."}),
                "bypass": ("BOOLEAN", {"default": False,
                    "tooltip": "When true, the node passes MODEL through unchanged. Use this to "
                               "disable IPAdapter for a specific gen without unwiring."}),
            },
            "optional": {
                # IMAGE is optional so a bypass=True flow doesn't need a dummy
                # reference wired. With bypass=False and no reference, we
                # raise a clear error rather than crashing inside the upstream.
                "reference": ("IMAGE", {"tooltip": "Character reference image. ONE high-quality canonical "
                                                     "shot of the character is the recommended source. "
                                                     "Required when bypass=False; can be unwired when bypass=True."}),
                "attn_mask": ("MASK", {"tooltip": "Optional attention mask for regional application — "
                                                    "the bridge to multi-panel single-gen workflows."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("MODEL with the IPAdapter character conditioning applied "
                        "(or passthrough if bypass=True). Wire into your sampler.",)
    FUNCTION = "anchor"
    CATEGORY = "GrimmRibbity/Character"

    def anchor(self, model, preset, weight, weight_type, start_at, end_at,
                bypass, reference=None, attn_mask=None):
        if bypass:
            return (model,)
        if reference is None:
            raise ValueError(
                "Character Anchor: bypass=False but no reference image is wired. "
                "Connect an IMAGE source to the 'reference' input, or set bypass=True "
                "to pass the model through unchanged."
            )

        loader = IPAdapterUnifiedLoader()
        model_with_ipa, ipadapter = loader.load_models(model, preset)

        applier = IPAdapterSimple()
        result = applier.apply_ipadapter(
            model_with_ipa, ipadapter, reference,
            weight=weight, start_at=start_at, end_at=end_at,
            weight_type=weight_type, attn_mask=attn_mask,
        )
        return result


NODE_CLASS_MAPPINGS = {"GrimmRibbityCharacterAnchor": GrimmRibbityCharacterAnchor}
NODE_DISPLAY_NAME_MAPPINGS = {"GrimmRibbityCharacterAnchor": "GrimmRibbity — Character Anchor"}
