"""Doctor node and the CrossDistill SIGMAS node."""

from __future__ import annotations

from ..schedules import crossdistill
from ..spark_runtime.doctor import format_report, run_doctor
from .common import CATEGORY, RUNTIME_TYPE


class SparkDiffusionDoctor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"runtime": (RUNTIME_TYPE,)},
            "optional": {"torch_compile_smoke_test": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "diagnose"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Check the SparkDiffusion runtime (GPU, CUDA, Triton, FP8, fused kernels, model files)."

    def diagnose(self, runtime, torch_compile_smoke_test=True):
        report = format_report(run_doctor(runtime, compile_smoke=bool(torch_compile_smoke_test)))
        print(report)
        return {"ui": {"text": [report]}, "result": (report,)}


class SparkDiffusionCrossDistillSigmas:
    """Exact upstream CrossDistill schedule as ComfyUI ``SIGMAS``.

    Matches upstream sampling when used with a flow model at shift 1.0, the
    Euler sampler and CFG 1.0. It is a schedule only: it does not provide RoLA
    sparse attention, so on its own it is *not* SparkDiffusion acceleration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (list(map(str, crossdistill.SUPPORTED_STEPS)), {"default": "4"}),
                "sigma_max": (
                    "FLOAT",
                    {"default": crossdistill.SIGMA_MAX_DEFAULT, "min": 1.0, "max": 1e6, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "build"
    CATEGORY = CATEGORY + "/experimental"
    DESCRIPTION = (
        "CrossDistill few-step schedule (exact upstream values) as SIGMAS for native samplers. "
        "Schedule only - no RoLA sparse attention."
    )

    def build(self, steps, sigma_max):
        import torch

        values = crossdistill.crossdistill_timesteps(int(steps), float(sigma_max))
        return (torch.tensor(values, dtype=torch.float32),)
