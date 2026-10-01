"""SparkDiffusionProfile: pick a released checkpoint and its exact upstream settings."""

from __future__ import annotations

import dataclasses
import json

from ..spark_runtime.profiles import DEFAULT_PROFILE_KEY, PROFILES
from .common import CATEGORY, PROFILE_TYPE


class SparkDiffusionProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(PROFILES),
                    {
                        "default": DEFAULT_PROFILE_KEY,
                        "tooltip": "Released SparkDiffusion checkpoint. Each preset pins resolution, step count, "
                        "CrossDistill schedule and RoLA top-k ratio.",
                    },
                ),
            },
            "optional": {
                "topk_ratio_override": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0 = profile default. RoLA keep ratio (0.1 = 90% sparsity, 0.05 = 95%, 0.03 = 97%); "
                        "must lie in the range the checkpoint supports.",
                    },
                ),
            },
        }

    RETURN_TYPES = (PROFILE_TYPE, "STRING")
    RETURN_NAMES = ("profile", "profile_info")
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = "Select a SparkDiffusion model profile (Wan2.1 / Wan2.2, resolution, steps, sparsity)."

    def select(self, preset, topk_ratio_override=0.0):
        profile = PROFILES[preset]
        if topk_ratio_override and topk_ratio_override > 0:
            topk = profile.validate_topk(round(float(topk_ratio_override), 6))
            profile = dataclasses.replace(profile, topk_ratio=topk)
        return (profile, json.dumps(profile.as_dict(), indent=2))
