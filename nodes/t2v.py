"""SparkDiffusionTextToVideo."""

from __future__ import annotations

import json

from ..spark_runtime.command import GenerationRequest
from ..spark_runtime.profiles import ASPECT_RATIOS
from .common import CATEGORY, MAX_SEED, PROFILE_TYPE, RUNTIME_TYPE, run_generation, ui_for, video_output
from .common import load_frames as decode_frames


class SparkDiffusionTextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": (RUNTIME_TYPE,),
                "profile": (PROFILE_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": "A cat playing in the garden under the sun."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True}),
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 5,
                        "max": 161,
                        "step": 4,
                        "tooltip": "4k+1 frames at 16 fps; 81 = 5 s (the length the checkpoints were distilled for).",
                    },
                ),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "16:9"}),
                "filename_prefix": ("STRING", {"default": "sparkdiffusion/SparkWan"}),
            },
            "optional": {
                "load_frames": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Decode every frame into the IMAGE output (memory heavy). Off = first frame only.",
                    },
                ),
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
    DESCRIPTION = (
        "Text-to-video with the official SparkDiffusion inference (RoLA sparse attention + CrossDistill "
        "few-step sampling) in an isolated runtime."
    )

    def generate(
        self, runtime, profile, prompt, seed, num_frames, aspect_ratio, filename_prefix, load_frames=False, num_videos=1
    ):
        if profile.task != "t2v":
            raise ValueError(f"{profile.label} is not a text-to-video profile; use the Image To Video node")
        req = GenerationRequest(
            prompt=prompt,
            seed=int(seed),
            num_frames=int(num_frames),
            aspect_ratio=aspect_ratio,
            num_samples=int(num_videos),
            topk_ratio=profile.topk_ratio,
        )
        result = run_generation(runtime, profile, req, filename_prefix)
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
