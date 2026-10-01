# Windows native backend (experimental)

Upstream SparkDiffusion documents Linux only. `backend = windows-native` runs the
**unmodified** upstream inference entry points in a Windows Python venv. It works
on RTX 5090, but two PyTorch-on-Windows gaps need small, clearly reported
compatibility shims, so this backend is marked **experimental**. For the exact
upstream code path on a Windows machine use [WSL2](WSL2.md).

## What we validated (RTX 5090, Windows 11, torch 2.14.1+cu130, triton-windows 3.7.1)

Validation order as run on the test machine:

| step | configuration | result |
|---|---|---|
| 1 | BF16, fused kernels **off** | not run: upstream's non-fused RoLA path needs the external thu-ml/SLA training kernel (`SLA_SRC`); the node refuses this combination without `sla_src` |
| 1' | BF16, fused kernels on, torch.compile on | works (1.3B 480P, valid video) |
| 2 | FP8 W8A8 | upstream path fails: Windows PyTorch has no row-wise `_scaled_mm`. Works with `fp8_compat=auto` (see below) |
| 3 | Triton | triton-windows 3.7.1 compiles and runs upstream's Triton kernels (FP8 GELU-quant, RoLA fused kernels) |
| 4 | fused kernels | active (verified per run, see metadata `fused_kernels_active`) |
| 5 | torch.compile | works; `cl.exe` was not on PATH, triton-windows' bundled TinyCC was enough |
| 5' | torch.compile **off** (eager) | works with the eager SDPA fallback below |

Profiles generated on Windows native with real weights: Wan2.1 T2V 1.3B 480P, 14B 480P, 14B 720P 3-step,
I2V 14B 720P and Wan2.2 A14B 480P, plus a ComfyUI run and a ComfyUI cancellation. Each run passed the
launcher's checks (RoLA, sparsity, CrossDistill timesteps, FP8) and ffprobe validation.
14B 720P 3-step warm denoise: 21.6 s compiled, 39.0 s eager ([BENCHMARK.md](BENCHMARK.md)).

## Gap 1: FP8 row-wise GEMM (`fp8_compat`)

Upstream quantizes weights per output channel and activations per token and
calls `torch._scaled_mm` with row-wise scales. Windows CUDA builds of PyTorch
(checked: 2.9.1+cu130 and 2.14.1+cu130) only ship tensor-wise scaled FP8 GEMMs.

With `fp8_compat = auto` the launcher installs a shim that runs the same FP8
operands through a tensor-wise GEMM (scales = 1) and applies the row-wise scales
and bias in an epilogue (fused into one pointwise kernel under torch.compile).
The FP8 weights, activations and scales are exactly upstream's; only *where* the
scales are applied changes. Checks we ran:

* unit test against a dequantized FP32 reference: relative error < 1 %
  (`tests/test_launcher.py::test_rowwise_compat_matches_dequantized_reference`);
* same seed, same platform (WSL2), eager: native row-wise vs compat PSNR 21.6 dB,
  while native eager vs native compiled differ by 19.7 dB. Few-step video models
  amplify tiny numeric differences; the compat path deviates from native *less*
  than switching torch.compile on or off does. Frames are visually equivalent.

The metadata records which GEMM path ran (`fp8_gemm_backend`). Set
`fp8_compat = off` to fail instead of emulating.

## Gap 2: no FlashAttention SDPA (eager mode only)

Upstream's eager attention asks PyTorch for `[FLASH, cuDNN, EFFICIENT]` via
`sdpa_kernel(..., set_priority_order=True)`. Recent PyTorch removed that keyword,
so upstream falls back to `[FLASH]` only, and Windows builds have no FlashAttention
SDPA ("No available kernel"). With torch.compile off, the launcher substitutes
memory-efficient SDPA (exact attention) in that one situation and records
`eager_sdpa_fallback` in the metadata. The compiled path is unaffected.

## Setup

```powershell
# anywhere outside ComfyUI, e.g. D:\AI\sparkdiffusion
uv venv venv-win --python 3.12              # or: py -3.12 -m venv venv-win
uv pip install --python venv-win\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python venv-win\Scripts\python.exe -r C:\ComfyUI\custom_nodes\ComfyUI-SparkDiffusion\requirements-runtime.txt
git clone https://github.com/AlibabaResearch/SparkDiffusion
```

`requirements-runtime.txt` selects `triton-windows` on Windows. Do not install
these packages into ComfyUI's own Python.

Runtime node: `backend = windows-native` (or `auto`), `python = D:\AI\sparkdiffusion\venv-win\Scripts\python.exe`,
`sparkdiffusion_repo = D:\AI\sparkdiffusion\SparkDiffusion`, `model_root = D:\AI\models\sparkdiffusion`.

## Notes

* The child process gets ComfyUI's environment minus `PYTHONPATH`,
  `PYTHONHOME`, `VIRTUAL_ENV` and similar, so ComfyUI's packages cannot leak
  into the runtime.
* The native Windows CUDA allocator does not support `expandable_segments`. The launcher therefore sets
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` on Windows (unless you set the variable yourself). Without
  it, 14B 720P steps took about 2x longer: VRAM use crossed 32 GB and the driver spilled into system memory.
  To get an error instead of a silent slowdown in that situation, set *CUDA - Sysmem Fallback Policy* to
  *Prefer No Sysmem Fallback* in the NVIDIA Control Panel.
* The Windows desktop shares VRAM with the runtime. On 32 GB cards close other
  GPU-heavy applications before running 14B 720P profiles.
