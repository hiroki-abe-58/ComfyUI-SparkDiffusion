# MLX / Apple Silicon roadmap

Status: **planned**. Nothing in v0.1 runs on Apple Silicon. SparkDiffusion's
inference stack (CUDA, Triton kernels, FP8 tensor cores) has no Metal
equivalent, so an MLX backend means porting the *algorithm*, not wrapping the
upstream code.

## Target

MacBook Pro M1 Max, 64 GB unified memory, **Wan2.1 T2V 1.3B 480P only**
(`SparkWan2.1-T2V-1.3B-480P-0.90Sparsity`). 14B profiles are out of scope for a
first feasibility study.

## Plan: reuse an MLX Wan, port only what SparkDiffusion adds

Existing MLX Wan 2.1 implementations to evaluate as the base (not yet evaluated
for this project):

* [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) (Wan2.1 T2V/I2V example)
* [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) (Wan2.1 1.3B / 14B T2V and I2V)

New code would be limited to:

1. **Checkpoint conversion**: SparkWan `.pth` (`net_ema.`/`net.` prefixes,
   plus RoLA `proj_q`, `proj_k`, `gate_proj`, `gate_bias`) to the base's
   weight layout.
2. **RoLA attention**: block-mean-pooled Q/K scores, top-k block selection
   (`topk_ratio`, 64x64 blocks), block-sparse softmax attention, the rank-64
   low-rank linear branch with truncated interleaved RoPE, and the sigmoid gate
   that mixes both (see upstream `sparkdiffusion/networks/wan_rola_attention.py`).
   A naive MLX version gathers the selected K/V blocks per query block; a Metal
   kernel would come later.
3. **CrossDistill scheduler**: already implemented and tested here
   (`schedules/crossdistill.py`, exact upstream knots).

## Correctness before speed

* Dense-attention fallback is allowed **only** to verify the port
  (compare against the CUDA reference output for the same seed). It must never
  be presented as "SparkDiffusion acceleration".
* Reference: per-step DiT outputs from a CUDA run of upstream for a fixed seed
  and prompt (a dump hook in the launcher would be the first step; it does not
  exist yet).

## Open questions

* BF16 matmul throughput on M1 Max and whether block gathering dominates at
  32,760 tokens (480P, 81 frames).
* umT5-XXL (11 GB in BF16) on 64 GB unified memory alongside the DiT and VAE:
  staged loading as on CUDA.
* No FP8 on Apple GPUs; the MLX path would be BF16 (or 8-bit MLX quantization,
  which is a different recipe from upstream FP8 W8A8).
