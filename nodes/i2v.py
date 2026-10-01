"""SparkDiffusionImageToVideo (Wan2.1 I2V 14B)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from ..spark_runtime.command import GenerationRequest
from .common import CATEGORY, MAX_SEED, PROFILE_TYPE, RUNTIME_TYPE, run_generation, temp_directory, ui_for, video_output
from .common import load_frames as decode_frames


def save_input_image(image, directory: str) -> str:
    """Write the first image of a ComfyUI ``IMAGE`` batch as a PNG for the upstream entry point."""
    import numpy as np
    from PIL import Image

    arr = image[0].detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image[0])
    arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path = os.path.join(directory, "input_image.png")
    Image.fromarray(arr[..., :3]).save(path)
    return path


class SparkDiffusionImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": (RUNTIME_TYPE,),
                "profile": (PROFILE_TYPE,),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "The subject slowly turns toward the camera."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True}),
                "num_frames": ("INT", {"default": 81, "min": 5, "max": 161, "step": 4}),
                "filename_prefix": ("STRING", {"default": "sparkdiffusion/SparkWanI2V"}),
            },
            "optional": {
                "fixed_resolution": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off = upstream input-aspect sizing (keeps the image aspect at the profile's "
                        "pixel area). "
                        "On = force the profile resolution at 16:9.",
                    },
                ),
                "load_frames": ("BOOLEAN", {"default": False}),
                "num_videos": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "tooltip": "Videos generated in ONE runtime process with seeds seed, seed+1, ... (upstream "
                        "NUM_SAMPLES). Loading and torch.compile are paid once, so extra videos are cheap.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "frames", "video_path", "metadata")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Image-to-video with the official SparkDiffusion Wan2.1 I2V inference in an isolated runtime."

    def generate(
        self,
        runtime,
        profile,
        image,
        prompt,
        seed,
        num_frames,
        filename_prefix,
        fixed_resolution=False,
        load_frames=False,
        num_videos=1,
    ):
        if profile.task != "i2v":
            raise ValueError(f"{profile.label} is not an image-to-video profile; use the Text To Video node")
        tmp = tempfile.mkdtemp(prefix="sparkdiffusion-i2v-", dir=temp_directory())
        try:
            image_path = save_input_image(image, tmp)
            req = GenerationRequest(
                prompt=prompt,
                seed=int(seed),
                num_frames=int(num_frames),
                aspect_ratio="16:9",
                num_samples=int(num_videos),
                topk_ratio=profile.topk_ratio,
                image_path=image_path,
                fixed_resolution=bool(fixed_resolution),
            )
            result = run_generation(runtime, profile, req, filename_prefix)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        video_path = result.video_paths[0]
        frames = decode_frames(video_path, bool(load_frames))
        return {
            "ui": ui_for(result.video_paths),
            "result": (
                video_output(video_path),
                frames,
                video_path,
                json.dumps(result.metadata, ensure_ascii=False, indent=2),
            ),
        }
