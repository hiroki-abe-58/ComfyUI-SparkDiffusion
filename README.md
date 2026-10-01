# ComfyUI-SparkDiffusion

Run [SparkDiffusion](https://github.com/AlibabaResearch/SparkDiffusion)'s sparse, few-step Wan video generation
from ComfyUI, with RTX 50-series / FP8 focused support.

[![CI](https://github.com/hiroki-abe-58/ComfyUI-SparkDiffusion/actions/workflows/test.yml/badge.svg)](https://github.com/hiroki-abe-58/ComfyUI-SparkDiffusion/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-lightgrey)
![Tested on RTX 5090](https://img.shields.io/badge/tested%20on-RTX%205090-76b900)

> This is an independent community integration and is not an official AlibabaResearch project.

ComfyUI-SparkDiffusion runs the **official, unmodified** SparkDiffusion inference entry points in an
**isolated runtime** (its own Python, possibly inside WSL2) and *verifies on every run* that the SparkDiffusion
path was actually used: RoLA sparse attention on every block, the exact CrossDistill timesteps, FP8 W8A8, fused
Triton kernels and torch.compile. Nothing is added to ComfyUI's own Python environment.

## Status

| platform / backend | status |
|---|---|
| RTX 5090 · **WSL2** (Ubuntu 24.04) · upstream FP8 | **Tested**: 1.3B, 14B 720P 3-step, 14B 720P 97 %; ComfyUI integration and cancellation |
| RTX 5090 · **Windows 11 native** · FP8 via compatibility shim | **Experimental**, tested: 1.3B, 14B 480P, 14B 720P 3-step, I2V 14B 720P, Wan2.2 A14B; ComfyUI integration and cancellation |
| Linux (bare metal) CUDA | Implemented, untested (same Linux code path as WSL2) |
| Other NVIDIA GPUs (Ada SM89, Hopper SM90) | Implemented, untested |
| ComfyUI | Tested with portable ComfyUI 0.36.0 (Windows), alongside 35 other custom nodes |
| Apple Silicon / MLX | Planned ([roadmap](docs/MLX_ROADMAP.md)) |

Measured on one RTX 5090 (see [Benchmark](#benchmark)): Wan2.1 14B 720P 3-step, **20.0 s** warm denoise
(WSL2, FP8, torch.compile) · Wan2.1 1.3B 480P 4-step, **1.78 s**.

Legend: **Tested** = generated and validated on real weights on the test machine;
**Implemented, untested** = wired up and covered by model-free tests, not run on real weights here;
**Experimental** = runs, but relies on a compatibility shim or an unvalidated path; **Planned** = not implemented.

## What SparkDiffusion is

SparkDiffusion (AlibabaResearch) accelerates Wan 2.1 / 2.2 video diffusion transformers by combining:

* **RoLA**: sparse low-rank attention. Each query block attends to its top-k key blocks (top-k *ratio* 0.1 / 0.05 / 0.03 =
  90 / 95 / 97 % sparsity) plus a rank-64 linear-attention branch, mixed by a learned gate.
* **CrossDistill**: few-step (3 or 4 steps), CFG-free distillation with a fixed timestep schedule.
* **FP8 W8A8**: row-wise FP8 for the FFN and all attention projections (`torch._scaled_mm`).
* **Fused Triton inference kernels** and **torch.compile**.

The SparkDiffusion paper reports up to 265x end-to-end acceleration on RTX 5090 under its benchmark configuration
(Wan2.1-T2V-14B-720P, 97 % sparsity, 3-step, versus the 50-step CFG dense baseline). That number is the paper's;
our own measurements are in [Benchmark](#benchmark) and [docs/BENCHMARK.md](docs/BENCHMARK.md).

Upstream: [repository](https://github.com/AlibabaResearch/SparkDiffusion) ·
[paper (arXiv:2609.23153)](https://arxiv.org/abs/2609.23153) ·
[RoLA (arXiv:2609.06712)](https://arxiv.org/abs/2609.06712) ·
[CrossDistill (arXiv:2609.14725)](https://arxiv.org/abs/2609.14725) ·
[weights (Hugging Face collection)](https://huggingface.co/collections/alibabagroup/sparkdiffusion) ·
[blog](https://sparkdiffusion.github.io/)

## How it works

```
ComfyUI ──> ComfyUI-SparkDiffusion nodes (no torch/triton imports at startup)
              │  job.json (UTF-8: prompt, argv, expected schedule) ── argument-list subprocess, never shell=True
              ▼
           isolated SparkDiffusion runtime  (Windows venv | WSL2 distro | Linux venv)
              launcher.py ── wraps upstream functions to observe & verify, then runpy(official entry point)
              ▼
           MP4 + metadata JSON (RoLA / schedule / FP8 / kernels / compile / timings / VRAM)
```

The launcher does **not** patch upstream source. It wraps the functions the entry point imports
(checkpoint loading, `optimize_model_for_inference`, umT5, VAE decode, video writer) to record and check:

| check | how it is verified |
|---|---|
| RoLA active | every transformer block's self-attention is `WanSelfAttentionRoLa`; RoLA weights loaded from the checkpoint (a checkpoint without them is refused instead of running random-init RoLA) |
| sparsity / top-k | the `rola_topk_ratio` of every RoLA layer equals the profile's |
| CrossDistill active | the timesteps actually fed to the DiT equal the CrossDistill schedule (bf16-exact) |
| FP8 active | number of `FP8Linear` modules after conversion (e.g. `FP8 active: 400 modules` for 14B) and which GEMM path ran |
| fused kernels | the RoLA layers' fused-inference flag and precision |
| torch.compile | the DiT is an Inductor `OptimizedModule` |

A run whose checks fail is reported as an error, not as a SparkDiffusion result.

## Supported models

| profile (node preset) | checkpoint | resolution | steps | sparsity | status |
|---|---|---|---|---|---|
| `wan2.1-t2v-1.3b-480p-s90-4step` | SparkWan2.1-T2V-1.3B-480P-0.90Sparsity | 480P | 4 | 90 % | **Tested** (WSL2, Windows) |
| `wan2.1-t2v-14b-480p-s90-4step` | SparkWan2.1-T2V-14B-480P-0.90Sparsity | 480P | 4 | 90 % | **Tested** (Windows) |
| `wan2.1-t2v-14b-720p-s95-3step` | SparkWan2.1-T2V-14B-720P-0.95Sparsity-3Step | 720P | 3 | 95 % (90–95 %) | **Tested** (WSL2, Windows) |
| `wan2.1-t2v-14b-720p-s97-4step` | SparkWan2.1-T2V-14B-720P-0.97Sparsity | 720P | 4 | 97 % (90–97 %) | **Tested** (WSL2) |
| `wan2.1-i2v-14b-720p-s97-4step` | SparkWan2.1-I2V-14B-720P-0.97Sparsity | 720P | 4 | 97 % (90–97 %) | **Tested** (Windows) |
| `wan2.2-t2v-a14b-480p-s95-4step` | SparkWan2.2-T2V-14B-480P-0.95Sparsity (High + Low) | 480P | 2+2 | 95 % (90–95 %) | **Tested** (Windows, single run) |

Ranges in brackets are the top-k overrides the model cards allow (Profile node `topk_ratio_override`).
"Tested" means a real generation on the RTX 5090 test machine produced a valid MP4 (frame count, resolution,
fps and duration checked with ffprobe), and the launcher verified RoLA on every block, the profile's sparsity, the
exact CrossDistill timesteps and FP8 for that run. Frames were also inspected visually.

All checkpoints are public (Apache-2.0) in the
[alibabagroup Hugging Face collection](https://huggingface.co/collections/alibabagroup/sparkdiffusion).
Shared Wan assets (VAE, umT5-XXL encoder, tokenizer; CLIP for I2V) come from the native `Wan-AI/Wan2.1-*` repositories.
The dense Wan DiT weights are **not** needed.

## Installation

### 1. The custom node (ComfyUI side)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-SparkDiffusion
```

No `pip install` is needed: the node has no Python dependencies beyond ComfyUI itself. Importing it does not import
torch, Triton, flash-attn or SparkDiffusion and loads no weights (checked by the test suite and CI).

### 2. The SparkDiffusion runtime

Pick one backend. Each is a separate Python environment that you create once.

| your machine | backend | guide |
|---|---|---|
| Windows + RTX 40/50, want upstream FP8 | `wsl2` (recommended) | [docs/WSL2.md](docs/WSL2.md) |
| Windows, no WSL | `windows-native` (experimental) | [docs/WINDOWS.md](docs/WINDOWS.md) |
| Linux | `linux` | below |

Linux:

```bash
python3.12 -m venv ~/sparkdiffusion/venv
~/sparkdiffusion/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
~/sparkdiffusion/venv/bin/pip install -r /path/to/ComfyUI/custom_nodes/ComfyUI-SparkDiffusion/requirements-runtime.txt
git clone https://github.com/AlibabaResearch/SparkDiffusion ~/sparkdiffusion/SparkDiffusion
```

`requirements-runtime.txt` mirrors upstream's `requirements.txt`, including **Triton >= 3.4**, which upstream needs
for native FP8 MMA lowering on RTX 50-series (SM120). flash-attn is optional at inference time.

### 3. Model placement

```
<model_root>/
├── Wan2.1-T2V-1.3B/                       Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/
├── Wan2.1-I2V-14B-720P/                   models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   (I2V only)
├── SparkWan2.1-T2V-1.3B-480P-0.90Sparsity/SparkWan2.1-T2V-1.3B-480P-0.90Sparsity.pth
└── SparkWan2.1-T2V-14B-720P-0.95Sparsity-3Step/SparkWan2.1-T2V-14B-720P-0.95Sparsity-3Step.pth
```

The VAE, umT5 and tokenizer files are byte-identical in every Wan 2.1 / 2.2 repository, so one copy serves all
profiles. Download what a profile needs (skips files you already have):

```bash
python scripts/download_models.py --profile wan2.1-t2v-1.3b-480p-s90-4step --model-root D:/AI/models/spark
python scripts/download_models.py --profile all --model-root /data/spark --dry-run   # only print hf commands
```

Sizes: Spark 1.3B checkpoint 2.8 GB, Spark 14B checkpoints 28.6 GB (I2V 32.8 GB), umT5 11.4 GB, VAE 0.5 GB, CLIP 4.8 GB.

### 4. Configure (no paths inside workflows)

Settings resolve in this order: **Runtime node field** > **environment variable** > **`sparkdiffusion_config.json`**
(git-ignored, next to this README; see `sparkdiffusion_config.example.json`).

| setting | env var |
|---|---|
| SparkDiffusion checkout | `SPARKDIFFUSION_REPO` |
| runtime interpreter | `SPARKDIFFUSION_PYTHON` |
| model root | `SPARKDIFFUSION_MODEL_ROOT` |
| backend | `SPARKDIFFUSION_BACKEND` |
| WSL distro | `SPARKDIFFUSION_WSL_DISTRO` |

Check everything before generating:

```bash
python scripts/doctor.py            # or add the "SparkDiffusion Doctor" node
```

```
[PASS] GPU                          NVIDIA GeForce RTX 5090
[PASS] Compute capability           sm_120
[PASS] FP8 W8A8 (row-wise)          torch._scaled_mm row-wise supported (upstream FP8 path)
[PASS] Triton                       3.8.0
[PASS] Fused kernels                sparkdiffusion.ops.fused_kernel imports
[PASS] torch.compile                smoke test OK
[PASS] Assets wan2.1-t2v-1.3b-480p-s90-4step checkpoint + Wan assets found
```

## Minimal workflow

Load `workflows/sparkdiffusion_1p3b_t2v.json` (drag it onto the ComfyUI canvas):

```
SparkDiffusion Runtime ─┐
                        ├─> SparkDiffusion Text To Video ─> (video preview, VIDEO, frames, path, metadata)
SparkDiffusion Profile ─┘
```

More: `workflows/sparkdiffusion_14b_720p_3step_t2v.json`, `workflows/sparkdiffusion_14b_720p_i2v.json`.

## Nodes

| node | purpose |
|---|---|
| **SparkDiffusion Runtime** | backend (`auto`, `windows-native`, `wsl2`, `linux`), interpreter, repo, model root, `quantization` (`fp8`/`bf16`), `fused_kernels`, `torch_compile`, `fp8_compat`, `sla_src`, timeout |
| **SparkDiffusion Profile** | released checkpoint preset (resolution, steps, schedule, top-k); optional top-k override within the range the checkpoint supports |
| **SparkDiffusion Text To Video** | prompt, seed, frames (4k+1, 81 = 5 s @ 16 fps), aspect ratio, filename prefix, `num_videos` (seeds seed, seed+1, ... in one process), `load_frames` → `VIDEO`, `IMAGE` frames, file path, metadata JSON; shows a video preview |
| **SparkDiffusion Image To Video** | same plus an input `IMAGE` and `fixed_resolution` (Wan2.1 I2V 14B) |
| **SparkDiffusion Doctor** | PASS/WARN/FAIL environment report from inside the runtime |
| **SparkDiffusion CrossDistill Sigmas** (experimental) | the exact CrossDistill schedule as `SIGMAS` for native samplers (flow model, shift 1.0, Euler, CFG 1). Schedule only: it does **not** give you RoLA sparse attention, so it is not SparkDiffusion acceleration by itself |

Each run writes `<prefix>_00001.mp4` and `<prefix>_00001.json` (metadata) into ComfyUI's output folder.
Cancelling in ComfyUI stops the runtime and its whole process tree (including inside WSL2).

## RTX 5090 / Blackwell (SM120)

* PyTorch must be a CUDA 12.8+ build (we use `cu130`); the doctor prints the compute capability (`sm_120`).
* **FP8**: upstream FP8 W8A8 requires row-wise `torch._scaled_mm`. Linux CUDA builds support it on SM120;
  **Windows CUDA builds do not**. Use `wsl2`, or `windows-native` with `fp8_compat=auto` (experimental emulation:
  tensor-wise FP8 GEMM + row-wise epilogue, same FP8 operands). FP8 converts 10 GEMMs per block:
  300 modules for 1.3B, 400 for 14B.
* **VRAM**: 14B at 720P peaks around 24 GiB allocated (28 GiB for I2V). The Windows desktop shares the card, so the
  launcher tunes PyTorch's allocator by default (`expandable_segments` on Linux/WSL2, `cudaMallocAsync` on
  Windows); without it, 14B steps ran about 2x slower here because VRAM spilled into system memory.
* **NVFP4**: not exposed. Upstream marks its NVFP4 path as a legacy backend for SM100 (B200/GB200), FFN-only and
  dependent on vLLM/FlashInfer/QuTLASS kernels. It is not the recommended RTX 5090 path; FP8 is.

### Windows 11 · WSL2 · Linux

| | Windows native | WSL2 | Linux |
|---|---|---|---|
| upstream FP8 (row-wise) | no (compat shim) | yes | yes |
| Triton / fused kernels | triton-windows | yes | yes |
| torch.compile | yes | yes | yes |
| FlashAttention SDPA (eager) | no (fallback shim) | yes | yes |
| model files | local | `/mnt/<drive>` or distro disk | local |

### fused kernels

On by default. Turning them off makes upstream use its non-fused RoLA path, which needs the external thu-ml/SLA
training kernel (`sla_src`); the node refuses that combination without `sla_src`.

### torch.compile

Upstream always compiles the DiT (`torch.compile`, mode `default`). The Runtime node makes it a toggle, because
every ComfyUI execution starts a fresh runtime process and pays the compile time again:

| 14B 720P 3-step, Windows native | compile off | compile on |
|---|---|---|
| first video in a process (denoise) | 54 s | 93–228 s (includes compiling) |
| each further video in the same process | 39 s | 21.6 s |

So `torch_compile` defaults to **off** for one-off runs. Turn it on together with `num_videos >= 2` (several seeds
in one process, upstream's `NUM_SAMPLES`), or for benchmarking. The metadata always records which mode ran.
The launcher also pre-registers einops' numpy backend: upstream's video writer otherwise forces a full DiT
recompile at the start of every second sample (about 25 s per sample at 1.3B).

## Benchmark

The SparkDiffusion paper reports up to 265x end-to-end acceleration on RTX 5090 under its benchmark configuration.
We did not reproduce that speedup (we did not run the 50-step dense baseline).

Our RTX 5090 test (single GPU, Windows 11 host; denoise = CUDA-synchronized DiT time, warm = 2nd/3rd video in one
process; full tables, VRAM per phase and telemetry in [docs/BENCHMARK.md](docs/BENCHMARK.md)):

| profile | backend | torch.compile | warm denoise | per step | VAE decode |
|---|---|---|---|---|---|
| 14B 720P 95 % 3-step | WSL2 | on | **20.0 s** | 6.7 s | 7.0 s |
| 14B 720P 95 % 3-step | Windows native | on | **21.6 s** | 7.3 s | 7.0 s |
| 14B 720P 95 % 3-step | Windows native | off | 39.0 s | 13.0 s | 7.1 s |
| 14B 720P 97 % 4-step | WSL2 | on | **24.5 s** | 6.1 s | 7.1 s |
| 1.3B 480P 90 % 4-step | WSL2 | on | **1.78 s** | 0.44 s | 3.0 s |
| 1.3B 480P 90 % 4-step | Windows native | on | **2.02 s** | 0.51 s | 3.0 s |

Environment: RTX 5090 32 GB (sm_120, driver 595.95), PyTorch 2.14.1+cu130, Triton 3.8.0 (WSL2) /
triton-windows 3.7.1, SparkDiffusion `ad6b65b`, FP8 W8A8, fused kernels on, RoLA and CrossDistill verified per run,
81 frames, fixed prompt and seed. Peak VRAM 23.4 GiB (umT5 load and VAE decode phases), 27.9 GiB for I2V 720P.
GPU during warm 14B denoise: 99 % utilization, about 560 W.

**Allocator note:** without allocator tuning, the same 14B runs took 39–48 s because VRAM use crossed 32 GB and the
Windows driver silently spilled to system memory. The launcher now sets `expandable_segments` (Linux/WSL2) or
`cudaMallocAsync` (Windows) by default. Details in [docs/BENCHMARK.md](docs/BENCHMARK.md#the-allocator-finding-32-gb-cards).

## Constraints and known limitations

* **One process per ComfyUI run.** Every run reloads umT5 (15–65 s) and, for 14B, the checkpoint plus FP8
  conversion (about 45 s from NTFS, about 3 min through WSL2's 9P). A persistent worker is the top roadmap item.
* **Windows native relies on two shims**: emulated row-wise FP8 GEMM and the eager SDPA fallback
  ([docs/WINDOWS.md](docs/WINDOWS.md)). Prefer WSL2 if you want the exact upstream code path.
* **Wan2.2 A14B** was validated with one run on Windows native (64 GB host RAM): both 14B experts are FP8-converted
  and swapped between CPU and GPU by upstream, so expect high system-RAM use. It was not run under WSL2, whose VM
  gets 50 % of host RAM by default. Its denoise time in our metadata covers the first to last DiT step; upstream's
  log line also includes moving the first expert to the GPU and the last one back.
* `fused_kernels = off` needs the external thu-ml/SLA training kernel (`sla_src`); this mode was not tested here.
* NVFP4 and upstream's legacy INT8 path are not exposed.
* Frame counts must be 4k+1; the checkpoints were distilled for 81 frames (5 s at 16 fps).
* I2V follows the input image's aspect ratio at the profile's pixel area (upstream behaviour) unless
  `fixed_resolution` is on.
* 14B profiles need about 24 GiB of free VRAM (28 GiB for I2V 720P). Cards with less than 32 GB were not tested.
* No native ComfyUI backend yet: models are not loaded into ComfyUI's model management, and LoRAs or ComfyUI
  samplers cannot be combined with RoLA. The SIGMAS node only exposes the schedule.
* Bare-metal Linux and non-Blackwell GPUs are implemented but were not run on this machine.

## Troubleshooting

| symptom | fix |
|---|---|
| `Rowwise scaling is not currently supported on your device` | Windows PyTorch build: use `backend=wsl2`, or `fp8_compat=auto`, or `quantization=bf16` |
| `No available kernel. Aborting execution.` with torch.compile off | handled automatically (eager SDPA fallback); if you run upstream scripts directly on Windows, enable torch.compile |
| `missing model files ...` | the message lists each missing file and its Hugging Face repo; run `scripts/download_models.py` |
| `RoLA parameters are missing from ...` | the checkpoint is not a SparkWan RoLA checkpoint for that profile |
| `SparkDiffusion verification failed: ...` | the run did not use the configured SparkDiffusion path; see the metadata `checks.problems` |
| very slow 14B steps, VRAM at 100 % | close other GPU apps; on Windows the driver silently spills to system RAM when VRAM is full |
| WSL2: runtime killed during model load | raise `memory=` in `%UserProfile%\.wslconfig` (see [docs/WSL2.md](docs/WSL2.md)) |
| `there is no distribution with the supplied name` | wrong `wsl_distro`; list with `wsl -l -v` |
| anything else | `python scripts/doctor.py`; error messages include backend, profile, exit code, command and log tail |

Errors never print your environment variables, and absolute paths in log tails and metadata are reduced to
`<path:filename>`.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q                     # model-free: schedules, profiles, paths, WSL conversion, subprocess, launcher, nodes
ruff check . && ruff format --check .
SPARKDIFFUSION_REPO=/path/to/SparkDiffusion pytest -q tests/test_schedules.py   # compare schedules with upstream source
python scripts/generate.py --profile wan2.1-t2v-1.3b-480p-s90-4step --prompt "..."   # CLI, same path as the nodes
python scripts/benchmark.py --profile wan2.1-t2v-14b-720p-s95-3step --torch-compile    # writes benchmarks/local/
```

## Roadmap

1. **Persistent runtime worker**: keep the compiled model resident between ComfyUI runs (removes the per-run load and
   compile cost, which dominates single runs today).
2. Broaden validation: run every profile on both Windows backends, bare-metal Linux, Wan2.2 under WSL2, and
   24 GB / Ada / Hopper GPUs.
3. Native ComfyUI backend (experimental, separate from the stable external runtime): checkpoint loader, RoLA attention,
   CrossDistill scheduler, FP8 path, ComfyUI model management/offload.
4. **MLX / Apple Silicon**: feasibility study for Wan2.1 1.3B 480P on M1 Max 64 GB; reuse an existing MLX Wan and port
   only RoLA, the CrossDistill scheduler and checkpoint conversion. See [docs/MLX_ROADMAP.md](docs/MLX_ROADMAP.md).

## Credits

* [AlibabaResearch/SparkDiffusion](https://github.com/AlibabaResearch/SparkDiffusion) (Apache-2.0): the models,
  inference code, kernels and papers this project runs. Please cite their work (below).
* Wan 2.1 / 2.2 by [Wan-AI](https://huggingface.co/Wan-AI) (Apache-2.0).
* Upstream itself acknowledges NVIDIA rCM, thu-ml SLA, Hugging Face finetrainers and Diffusers.

This repository does not vendor or modify upstream code. The CrossDistill schedule constants and profile settings
are transcribed from upstream (commit `ad6b65b`) and checked against it by the test suite.

```bibtex
@misc{liu2026sparkdiffusionmitigatinghighsparsitytrap,
  title={SparkDiffusion: Mitigating the High-Sparsity Trap --- A Unified Framework for up to $265\times$ Single-GPU Acceleration of Visual Generation},
  author={Yuxi Liu and Haoyu Li and Zekun Zhang and Tengxu Sun and Yixiang Cai and Jiayong Li and Yifei Xia and Tianle Liu and Baole Ai and Ang Wang and Jiamang Wang and Lin Qu and Kai Zhang and Kun Yuan and Bin Cui},
  year={2026}, eprint={2609.23153}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.23153},
}
```

## License

[Apache-2.0](LICENSE). Model weights, upstream code and generated content have their own licenses and terms.
