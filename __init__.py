"""ComfyUI-SparkDiffusion: run SparkDiffusion (AlibabaResearch) from ComfyUI.

Independent community integration; not an official AlibabaResearch project.
Importing this package is lightweight: SparkDiffusion, Triton, flash-attn and
model weights are only touched inside the separate runtime process.
"""

if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # imported as a plain module (e.g. by pytest's directory collection), not as a ComfyUI custom node
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
