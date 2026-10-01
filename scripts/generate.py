"""Generate a video from the command line through the same path the nodes use.

Example::

    python scripts/generate.py --profile wan2.1-t2v-1.3b-480p-s90-4step \
        --prompt "A cat playing in the garden under the sun." --output-dir ./outputs
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap


def main() -> int:
    profiles = _bootstrap.load("spark_runtime.profiles")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _bootstrap.add_runtime_args(p)
    p.add_argument("--profile", default=profiles.DEFAULT_PROFILE_KEY, choices=sorted(profiles.PROFILES))
    p.add_argument("--prompt", default="")
    p.add_argument("--prompt-file", default="", help="UTF-8 text file with the prompt (overrides --prompt)")
    p.add_argument("--image", default="", help="input image (I2V profiles)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--aspect-ratio", default="16:9", choices=list(profiles.ASPECT_RATIOS))
    p.add_argument("--num-samples", type=int, default=1, help="sequential samples in one process (1st = warmup)")
    p.add_argument("--topk-ratio", type=float, default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--prefix", default="sparkdiffusion")
    p.add_argument("--no-strict", action="store_true", help="do not fail on verification problems")
    args = p.parse_args()

    command = _bootstrap.load("spark_runtime.command")
    runner = _bootstrap.load("spark_runtime.runner")
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()
    cfg = _bootstrap.runtime_from_args(args)
    profile = profiles.get_profile(args.profile)
    req = command.GenerationRequest(
        prompt=prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        aspect_ratio=args.aspect_ratio,
        num_samples=args.num_samples,
        topk_ratio=args.topk_ratio,
        image_path=args.image or None,
    )
    result = runner.generate(
        cfg,
        profile,
        req,
        args.output_dir,
        filename_prefix=args.prefix,
        strict_checks=not args.no_strict,
        on_line=lambda _s, line: print(line, flush=True),
    )
    print("\n" + result.summary)
    for vp in result.video_paths:
        print("video:", vp)
    print("metadata:", result.metadata_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
