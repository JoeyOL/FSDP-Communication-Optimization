"""Download GPT-2 (small) and export to ./models/gpt2 in the same format used by this repo.

This script downloads Hugging Face 'gpt2' (a.k.a GPT-2 small, ~124M) and saves:
- model.safetensors
- config.json
- generation_config.json (if available)
- tokenizer files (tokenizer.json, vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json)

It overwrites files under ./models/gpt2.

Usage:
  # Online (default): downloads from HF Hub then exports to ./models/gpt2
  python download_gpt2_small_to_models_gpt2.py

  # Offline: reuse local HF cache only (won't hit the network)
  python download_gpt2_small_to_models_gpt2.py --local_files_only

  # Offline: export from an existing local model directory
  python download_gpt2_small_to_models_gpt2.py --source_dir /path/to/local/gpt2

Optional:
  HF_HOME=/path/to/cache python download_gpt2_small_to_models_gpt2.py
  TRANSFORMERS_OFFLINE=1 python download_gpt2_small_to_models_gpt2.py  # if already cached
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download/export GPT-2 (small) into ./models/gpt2 (repo-compatible format)."
    )
    p.add_argument(
        "--repo_id",
        default="gpt2",
        help="Hugging Face repo id to download (default: gpt2).",
    )
    p.add_argument(
        "--source_dir",
        default=None,
        help="If set, export from this local directory instead of downloading from HF Hub.",
    )
    p.add_argument(
        "--out_dir",
        default="./models/gpt2",
        help="Output directory (default: ./models/gpt2).",
    )
    p.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use local HF cache; do not attempt any network access.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    src = args.source_dir or args.repo_id
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer safetensors when available.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            src,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            src,
            use_fast=True,
            local_files_only=args.local_files_only,
        )
    except OSError as e:
        offline = bool(args.local_files_only) or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        hint_lines = [
            "Failed to load GPT-2 files.",
            f"source: {src}",
            f"out_dir: {out_dir.resolve()}",
            "",
        ]
        if offline:
            hint_lines += [
                "You're in offline/local-only mode, but the files were not found in the local cache.",
                "Options:",
                "  1) Copy a previously downloaded GPT-2 directory here and re-run with --source_dir <dir>",
                "  2) Disable offline mode and allow network access (unset TRANSFORMERS_OFFLINE, omit --local_files_only)",
            ]
        else:
            hint_lines += [
                "Network might be unavailable, or HF Hub is blocked.",
                "Options:",
                "  1) Try again with --local_files_only if you already have HF cache on this machine",
                "  2) Download on a machine with internet, then copy the model dir and use --source_dir",
            ]
        raise RuntimeError("\n".join(hint_lines)) from e

    # Save in the repo's expected structure.
    tokenizer.save_pretrained(out_dir)

    # save_pretrained(..., safe_serialization=True) writes model.safetensors
    model.save_pretrained(out_dir, safe_serialization=True)

    # Small sanity print.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved '{src}' to: {out_dir.resolve()}")
    print(f"Parameter count: {n_params:,}")
    print("Files:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name} ({p.stat().st_size/1024/1024:.2f} MiB)")


if __name__ == "__main__":
    # Help users understand caches/offline mode quickly.
    if os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        print("TRANSFORMERS_OFFLINE=1 detected: transformers will only use local cache.")
    main()
