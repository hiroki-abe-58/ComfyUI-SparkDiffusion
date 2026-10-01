# Benchmarks

## Paper claims (not reproduced here)

The SparkDiffusion paper reports up to 265x end-to-end acceleration on RTX 5090 under its benchmark configuration:
Wan2.1-T2V-14B-720P, 97 % sparsity, 3-step CFG-free, versus the 50-step CFG dense baseline (4769 s -> 18 s).
It also reports 1.3 s for Wan2.1-T2V-1.3B-480P (90 %, 3-step) on RTX 5090. These are upstream's numbers. We did
not run the dense 50-step baseline, so we make **no speedup claim** of our own.

## Our RTX 5090 tests

Measured with `scripts/benchmark.py` (fixed prompt and seed, 81 frames, 16:9). Each row is one runtime process
that generated 3 videos sequentially, as upstream's `NUM_SAMPLES=3` does. The first sample (**warmup**) pays Triton
autotuning and torch.compile. **Denoise** is CUDA-synchronized DiT time for all steps, the same quantity upstream logs
as "denoising time"; VAE decode and video writing are reported separately.

**Environment**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, sm_120, driver 595.95 |
| OS | Windows 11 Pro (26200); WSL2 kernel 6.6.87.2, Ubuntu 24.04 |
| PyTorch / CUDA | 2.14.1+cu130 / 13.0 (both backends) |
| Triton | 3.8.0 (WSL2) · triton-windows 3.7.1 (Windows) |
| SparkDiffusion | `ad6b65b34f89a2da59fde3238a6a750f1ed19529` |
| FP8 GEMM | WSL2: upstream row-wise `_scaled_mm` · Windows: tensor-wise + row-wise epilogue (`fp8_compat=auto`) |
| fused kernels | on · RoLA active on every block · CrossDistill timesteps verified on every run |

### Headline: Wan2.1 T2V 14B 720P, 95 % sparsity, 3-step (`wan2.1-t2v-14b-720p-s95-3step`)

| backend | torch.compile | warmup denoise | warm denoise (s/step) | VAE decode | GPU power (denoise) |
|---|---|---|---|---|---|
| WSL2 | on | 176.2 s | **20.0 s / 20.4 s** (6.7) | 7.0 s | 563 W avg, 99 % util |
| Windows native | on | 92.9 s ¹ | **21.6 s / 21.9 s** (7.3) | 7.0–7.3 s | 555 W avg, 98 % util |
| Windows native | off | 54.2 s | 39.0 s (13.0) | 7.1 s | 547 W avg |

¹ Inductor's on-disk cache was already warm from an earlier Windows run; a cold first compile took 228 s.

Peak VRAM (torch, allocated): 23.4 GiB.

### Wan2.1 T2V 14B 720P, 97 % sparsity, 4-step (`wan2.1-t2v-14b-720p-s97-4step`)

| backend | torch.compile | warmup denoise | warm denoise (s/step) | VAE decode |
|---|---|---|---|---|
| WSL2 | on | 165.4 s | **24.5 s / 24.6 s** (6.1) | 7.1 s |

At 6.1 s per step, a 3-step run at 97 % sparsity would take about 18.4 s. That is in line with the paper's 18 s,
but it is an extrapolation: the released 97 % checkpoint is a 4-step model.

### Smoke: Wan2.1 T2V 1.3B 480P, 90 % sparsity, 4-step (`wan2.1-t2v-1.3b-480p-s90-4step`)

| backend | torch.compile | warmup denoise | warm denoise (s/step) | VAE decode |
|---|---|---|---|---|
| WSL2 | on | 122.5 s | **1.78 s / 1.78 s** (0.44) | 3.0 s |
| Windows native | on | 56.6 s | **2.07 s / 2.02 s** (0.51) | 3.0 s |

### Single runs (one video per process, torch.compile off)

| profile | backend | denoise | VAE decode | total incl. loading | peak VRAM |
|---|---|---|---|---|---|
| 14B 480P 90 % 4-step | Windows native | 23.6 s | 3.3 s | 97.7 s | 23.4 GiB |
| I2V 14B 720P 97 % 4-step (1248x720) | Windows native | 61.7 s | 17.5 s | 182.3 s | 27.9 GiB |
| Wan2.2 A14B 480P 95 % 2+2-step | Windows native | 35.5 s ² | 3.3 s | 232.1 s | 23.4 GiB |
| 1.3B 480P 90 % 4-step, through ComfyUI | Windows native | n/a | n/a | 38.3 s | n/a |

² First to last DiT step, including the high-to-low expert swap. Upstream's own log line (57.9 s) also counts
moving the high-noise expert to the GPU before step 1 and the low-noise expert back to the CPU after step 4.

## What dominates a single ComfyUI run

Each ComfyUI execution starts a fresh runtime process, so it pays:

* umT5-XXL load and encode: 15 s (Windows, NTFS) to 65 s (WSL2 reading from `/mnt/e`);
* checkpoint load and FP8 conversion for 14B: 43–47 s (Windows) to 170–200 s (WSL2 through 9P);
* torch.compile: 50–230 s, only paid when compile is on.

That is why the node defaults to `torch_compile = off`. Use compile together with `num_videos >= 2`, which runs
upstream's sequential `NUM_SAMPLES`, when you generate several videos.

## The allocator finding (32 GB cards)

The first WSL2 headline run took **39–40 s** warm instead of 20 s. The GPU drew only ~380 W, nvidia-smi showed
31.9 of 32.6 GB in use, and PyTorch had reserved 29.6 GiB while allocating only 21.6 GiB. Upstream's text-encoder
and FP8-staging phases leave the caching allocator fragmented. Together with the Windows desktop's own VRAM use, that
pushed total usage past 32 GB, and the Windows driver (also under WSL2) silently spilled to system memory. The
Windows-native run showed the same effect (44.6–48.4 s).

The launcher now sets `PYTORCH_CUDA_ALLOC_CONF` to `expandable_segments:True` on Linux/WSL2 and to
`backend:cudaMallocAsync` on Windows (the native Windows allocator has no expandable segments), unless you set the
variable yourself. Allocator choice does not change numerics.

| | before | after |
|---|---|---|
| WSL2 warm denoise | 39.0–40.1 s | 20.0–20.4 s |
| Windows warm denoise | 44.6–48.4 s | 21.6–21.9 s |
| VAE decode | 10–16 s | 7.0–7.3 s |

## VRAM residency (14B 720P, per phase, torch allocated)

| phase | peak |
|---|---|
| umT5 encode (umT5 then unloaded: 1.1 GiB left) | 23.4 GiB |
| FP8 conversion + DiT to GPU (FP8 14B DiT resident) | 13.8 GiB |
| denoise | 21.3 GiB |
| VAE decode (DiT stays resident, as upstream does) | 23.4 GiB |

The checkpoint stays memory-mapped on the CPU and is converted to FP8 one transformer block at a time
(upstream's staged quantization), so the BF16 14B model never has to fit on the GPU.

## Reproduce

```bash
python scripts/benchmark.py --profile wan2.1-t2v-14b-720p-s95-3step --backend wsl2 --torch-compile --samples 3
```

Records (JSON + Markdown, no absolute paths) go to `benchmarks/local/`. That folder is git-ignored and includes
nvidia-smi telemetry (memory, utilization, power, temperature) and NVML samples per denoising window.
