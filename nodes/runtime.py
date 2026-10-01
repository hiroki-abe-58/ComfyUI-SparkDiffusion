"""SparkDiffusionRuntime: where and how the isolated SparkDiffusion runtime runs."""

from __future__ import annotations

from ..spark_runtime import config as C
from ..spark_runtime.command import FP8_COMPAT_MODES, RuntimeConfig
from ..spark_runtime.paths import BACKENDS
from .common import CATEGORY, RUNTIME_TYPE


class SparkDiffusionRuntime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (
                    list(BACKENDS),
                    {
                        "default": "auto",
                        "tooltip": "auto = windows-native on Windows, linux elsewhere. wsl2 runs the runtime inside "
                        "a WSL2 "
                        "distro (recommended for FP8 on Windows).",
                    },
                ),
                "quantization": (
                    ["fp8", "bf16"],
                    {
                        "default": "fp8",
                        "tooltip": "fp8 = upstream W8A8 FP8 (FFN + attention projections). bf16 = no quantization.",
                    },
                ),
                "fused_kernels": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Upstream Triton inference kernels for RoLA. Disabling requires an external "
                        "thu-ml/SLA "
                        "checkout (sla_src).",
                    },
                ),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "torch.compile the DiT (upstream default). Faster steps, but each ComfyUI run "
                        "starts a fresh "
                        "process and pays the compile time again.",
                    },
                ),
            },
            "optional": {
                "sparkdiffusion_repo": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "SparkDiffusion checkout. Empty = SPARKDIFFUSION_REPO or "
                        "sparkdiffusion_config.json.",
                    },
                ),
                "python": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Runtime interpreter (its own venv). For wsl2 this is a Linux path inside "
                        "the distro.",
                    },
                ),
                "model_root": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Directory holding the Wan assets and SparkWan checkpoints (see README).",
                    },
                ),
                "wsl_distro": ("STRING", {"default": "", "tooltip": "WSL distro name (wsl2 backend only)."}),
                "fp8_compat": (
                    list(FP8_COMPAT_MODES),
                    {
                        "default": "auto",
                        "tooltip": "When PyTorch lacks row-wise FP8 GEMMs (Windows builds), auto emulates them with a "
                        "tensor-wise GEMM + epilogue (experimental). off = fail instead.",
                    },
                ),
                "sla_src": ("STRING", {"default": "", "tooltip": "thu-ml/SLA checkout (only for fused_kernels=off)."}),
                "timeout_seconds": ("INT", {"default": 3600, "min": 0, "max": 86400, "tooltip": "0 = no timeout"}),
            },
        }

    RETURN_TYPES = (RUNTIME_TYPE,)
    RETURN_NAMES = ("runtime",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Configure the isolated SparkDiffusion runtime (backend, interpreter, model root, FP8, kernels)."

    def build(
        self,
        backend,
        quantization,
        fused_kernels,
        torch_compile,
        sparkdiffusion_repo="",
        python="",
        model_root="",
        wsl_distro="",
        fp8_compat="auto",
        sla_src="",
        timeout_seconds=3600,
    ):
        file_cfg = C.load_config_file()
        backend_value = backend if backend != "auto" else (C.resolve("backend", None, file_cfg) or "auto")
        cfg = RuntimeConfig(
            backend=backend_value,
            repo=C.resolve("repo", sparkdiffusion_repo, file_cfg),
            python=C.resolve("python", python, file_cfg),
            model_root=C.resolve("model_root", model_root, file_cfg),
            wsl_distro=C.resolve("wsl_distro", wsl_distro, file_cfg),
            quantization=quantization,
            fp8_compat=fp8_compat,
            fused_kernels=bool(fused_kernels),
            torch_compile=bool(torch_compile),
            sla_src=C.resolve("sla_src", sla_src, file_cfg),
            timeout_s=int(timeout_seconds),
        )
        cfg.validate()
        return (cfg,)
