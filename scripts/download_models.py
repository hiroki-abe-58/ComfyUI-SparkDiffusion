"""Download exactly the files a SparkDiffusion profile needs into a model root.

    python scripts/download_models.py --profile wan2.1-t2v-1.3b-480p-s90-4step --model-root D:\\AI\\models\\spark
    python scripts/download_models.py --profile all --model-root /data/spark --dry-run

Layout produced (what the Runtime node's ``model_root`` expects)::

    <model_root>/Wan2.1-T2V-1.3B/   Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/*
    <model_root>/Wan2.1-I2V-14B-720P/   models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (I2V)
    <model_root>/<SparkWan repo>/   <checkpoint>.pth

Only the shared Wan assets (VAE, umT5, tokenizer and, for I2V, CLIP) are fetched,
once, and files already present under the model root are skipped. The dense Wan
DiT weights are not needed.
All repositories are public. If a token is ever required, use ``hf auth login``
or the ``HF_TOKEN`` environment variable; never put tokens in files.
"""

from __future__ import annotations

import argparse
import os
import sys

import _bootstrap

TOKENIZER_FILES = ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"]


# VAE, umT5 and tokenizer are byte-identical (same sha256) in every Wan 2.1 / 2.2 repository,
# so they are fetched once; CLIP comes from the I2V repository.
SHARED_SOURCE = ("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1-T2V-1.3B")
CLIP_SOURCE = ("Wan-AI/Wan2.1-I2V-14B-720P", "Wan2.1-I2V-14B-720P")


def plan(profile, profiles_mod, model_root: str):
    """(repo, subdir, files) still missing under ``model_root`` for ``profile``."""
    paths = _bootstrap.load("spark_runtime.paths")
    have = paths.resolve_assets(model_root, profile) if os.path.isdir(model_root) else None
    jobs = []
    if have is None or not (have.vae and have.t5 and have.tokenizer):
        shared = [profiles_mod.VAE_FILE, profiles_mod.T5_FILE]
        shared += [f"{profiles_mod.TOKENIZER_DIR}/{f}" for f in TOKENIZER_FILES]
        jobs.append((*SHARED_SOURCE, shared))
    if profile.needs_clip and (have is None or not have.clip):
        jobs.append((*CLIP_SOURCE, [profiles_mod.CLIP_FILE]))
    if have is None or len(have.checkpoints) != len(profile.checkpoint_files):
        jobs.append((profile.hf_repo, profile.repo_dirname, list(profile.checkpoint_files)))
    return jobs


def main() -> int:
    profiles = _bootstrap.load("spark_runtime.profiles")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", required=True, choices=sorted(profiles.PROFILES) + ["all"])
    p.add_argument("--model-root", required=True)
    p.add_argument("--dry-run", action="store_true", help="print the hf download commands only")
    args = p.parse_args()

    selected = list(profiles.PROFILES.values()) if args.profile == "all" else [profiles.get_profile(args.profile)]
    jobs: dict = {}  # (repo, subdir) -> files, merged across profiles, order preserved
    for prof in selected:
        for repo, sub, files in plan(prof, profiles, args.model_root):
            merged = jobs.setdefault((repo, sub), [])
            merged.extend(f for f in files if f not in merged)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    for (repo, sub), files in jobs.items():
        target = os.path.join(args.model_root, sub)
        if args.dry_run or hf_hub_download is None:
            print(f'hf download {repo} {" ".join(files)} --local-dir "{target}"')
            continue
        for f in files:
            print(f"[download] {repo} :: {f}", flush=True)
            hf_hub_download(repo_id=repo, filename=f, local_dir=target)
    if hf_hub_download is None and not args.dry_run:
        print("\nhuggingface_hub is not installed here; run the commands above (pip install -U huggingface_hub).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
