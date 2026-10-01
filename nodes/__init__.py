"""ComfyUI node classes for ComfyUI-SparkDiffusion."""

from .extras import SparkDiffusionCrossDistillSigmas, SparkDiffusionDoctor
from .i2v import SparkDiffusionImageToVideo
from .profile import SparkDiffusionProfile
from .runtime import SparkDiffusionRuntime
from .t2v import SparkDiffusionTextToVideo

NODE_CLASS_MAPPINGS = {
    "SparkDiffusionRuntime": SparkDiffusionRuntime,
    "SparkDiffusionProfile": SparkDiffusionProfile,
    "SparkDiffusionTextToVideo": SparkDiffusionTextToVideo,
    "SparkDiffusionImageToVideo": SparkDiffusionImageToVideo,
    "SparkDiffusionDoctor": SparkDiffusionDoctor,
    "SparkDiffusionCrossDistillSigmas": SparkDiffusionCrossDistillSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkDiffusionRuntime": "SparkDiffusion Runtime",
    "SparkDiffusionProfile": "SparkDiffusion Profile",
    "SparkDiffusionTextToVideo": "SparkDiffusion Text To Video",
    "SparkDiffusionImageToVideo": "SparkDiffusion Image To Video",
    "SparkDiffusionDoctor": "SparkDiffusion Doctor",
    "SparkDiffusionCrossDistillSigmas": "SparkDiffusion CrossDistill Sigmas (experimental)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
