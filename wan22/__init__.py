"""GrimmRibbity Wan2.2 i2v 14B subpackage.

Three convenience nodes wrapping ComfyUI core for the two-expert Wan2.2 i2v
pipeline:
    - Loader: high_noise + low_noise UNETs + Wan2.2 VAE + UMT5 CLIP in one node.
    - Two-Stage KSampler: chains the two experts at a configurable step boundary.
    - Prompt Builder: shared PromptLibrary wildcard engine + motion-vocab presets.

I2V conditioning is intentionally NOT wrapped — use the core WanImageToVideo /
Wan22ImageToVideoLatent nodes (in comfy_extras.nodes_wan) between the loader
and sampler. They already handle start-image → concat-latent + temporal mask.
"""

from .loader import GrimmRibbityWan22I2VLoader
from .sampler import GrimmRibbityWan22TwoStageKSampler
from .prompt import GrimmRibbityWan22PromptBuilder


NODE_CLASS_MAPPINGS = {
    "GrimmRibbityWan22I2VLoader": GrimmRibbityWan22I2VLoader,
    "GrimmRibbityWan22TwoStageKSampler": GrimmRibbityWan22TwoStageKSampler,
    "GrimmRibbityWan22PromptBuilder": GrimmRibbityWan22PromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrimmRibbityWan22I2VLoader": "GrimmRibbity — Wan2.2 I2V Loader (14B)",
    "GrimmRibbityWan22TwoStageKSampler": "GrimmRibbity — Wan2.2 Two-Stage KSampler",
    "GrimmRibbityWan22PromptBuilder": "GrimmRibbity — Wan2.2 Prompt Builder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
