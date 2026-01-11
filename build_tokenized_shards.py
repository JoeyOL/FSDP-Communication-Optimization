"""Build tokenized shards for Wikipedia JSON (streaming) ahead of training.

This extracts the streaming tokenization + shard caching logic from `WikipediaDataset`
so you can pre-generate shards once and reuse them across training runs.

Input format: a JSON array: [ {"id":..., "title":..., "text":...}, ... ]

Outputs (by default in the same directory as the dataset):
- tokenized_<basename>_len<max_length>.meta.pt
- tokenized_<basename>_len<max_length>.shardXXXXXX.pt  (list[dict] encodings as produced by tokenizer)

Features:
- streaming parse with ijson (no OOM)
- shard writing
- resumable via meta.pt (processed_samples + finalized)

Typical:
  python build_tokenized_shards.py

Offline / deterministic advice:
- Run this once before `torchrun` training.
- Keep `max_length` and `shard_size` consistent with training args.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ijson
import torch
from tqdm import tqdm

from create_model import load_tokenizer
from logger import logger


def _atomic_save(obj: object, path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _is_compatible_meta(meta: dict, data_path: str, max_length: int, shard_size: int) -> bool:
    return (
        meta.get("data_path") == str(data_path)
        and int(meta.get("max_length", -1)) == int(max_length)
        and int(meta.get("shard_size", -1)) == int(shard_size)
    )


def _compat_mismatch_detail(meta: dict, data_path: str, max_length: int, shard_size: int) -> str:
    pairs = [
        ("data_path", str(data_path), str(meta.get("data_path"))),
        ("max_length", str(int(max_length)), str(meta.get("max_length"))),
        ("shard_size", str(int(shard_size)), str(meta.get("shard_size"))),
    ]
    diffs = []
    for k, cur, old in pairs:
        if cur != old:
            diffs.append(f"- {k}: current={cur} | meta={old}")
    if not diffs:
        return "(fields look equal, but meta was still considered incompatible)"
    return "\n".join(diffs)


def build_shards(
    *,
    data_path: str,
    max_length: int,
    shard_size: int,
    cache_dir: str | None,
    max_samples: int,
    force_restart: bool,
) -> str:
    tokenizer = load_tokenizer()

    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(data_path))
    os.makedirs(cache_dir, exist_ok=True)

    cache_prefix = os.path.join(cache_dir, f"tokenized_{os.path.basename(data_path)}_len{max_length}")
    cache_meta_path = f"{cache_prefix}.meta.pt"

    logger.info(f"Building tokenized shards (streaming) for: {data_path}")
    logger.info(f"cache_prefix: {cache_prefix}")

    shard_paths: list[str] = []
    shard_encodings: list[dict] = []
    resume_from = 0

    if force_restart and os.path.exists(cache_meta_path):
        logger.warning(f"force_restart=1: removing existing meta: {cache_meta_path}")
        os.remove(cache_meta_path)

    if os.path.exists(cache_meta_path):
        meta = torch.load(cache_meta_path)
        if not _is_compatible_meta(meta, data_path, max_length, shard_size):
            raise ValueError(
                "Existing meta is incompatible with current args:\n"
                f"{cache_meta_path}\n"
                f"Diffs:\n{_compat_mismatch_detail(meta, data_path, max_length, shard_size)}\n"
                "Fix: delete old cache or run with --force_restart."
            )
        if not meta.get("finalized", True):
            shard_paths = list(meta.get("shards", []))
            resume_from = int(meta.get("processed_samples", meta.get("num_samples", 0)))
            logger.info(
                f"Resuming unfinished cache: processed={resume_from}, shards={len(shard_paths)}"
            )
        else:
            logger.info(f"Cache already finalized: {cache_meta_path}")
            return cache_meta_path

    meta = {
        "data_path": str(data_path),
        "max_length": int(max_length),
        "shard_size": int(shard_size),
        "processed_samples": int(resume_from),
        "num_samples": int(resume_from),
        "shards": shard_paths,
        "finalized": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_save(meta, cache_meta_path)

    n = 0
    with open(data_path, "rb") as f:
        items = ijson.items(f, "item")
        pbar = tqdm(desc="tokenize(stream)", unit="sample")
        for idx, item in enumerate(items):
            if idx < resume_from:
                continue

            text = item.get("title", "") + "\n" + item.get("text", "")
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            shard_encodings.append(encoding)
            n += 1
            pbar.update(1)

            if shard_size > 0 and (n % shard_size == 0):
                shard_path = f"{cache_prefix}.shard{len(shard_paths):06d}.pt"
                torch.save(shard_encodings, shard_path)
                shard_paths.append(shard_path)
                shard_encodings = []

                meta["processed_samples"] = int(resume_from + n)
                meta["num_samples"] = int(resume_from + n)
                meta["shards"] = shard_paths
                _atomic_save(meta, cache_meta_path)

            if max_samples and max_samples > 0 and n >= max_samples:
                break

        pbar.close()

    if shard_encodings:
        shard_path = f"{cache_prefix}.shard{len(shard_paths):06d}.pt"
        torch.save(shard_encodings, shard_path)
        shard_paths.append(shard_path)

    total = int(resume_from + n)
    meta["processed_samples"] = total
    meta["num_samples"] = total
    meta["shards"] = shard_paths
    meta["finalized"] = True
    meta["finalized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_save(meta, cache_meta_path)

    logger.info(f"Done. meta: {cache_meta_path}")
    logger.info(f"Samples: {total}, shards: {len(shard_paths)}")
    return cache_meta_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-tokenize Wikipedia JSON into shard cache (.pt files).")
    p.add_argument(
        "--data_path",
        default="datasets/wikipedia_en_500mb.json",
        help="Path to wikipedia json array file.",
    )
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--shard_size", type=int, default=2000)
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to write shards/meta (default: same dir as data file).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only process first N new samples (useful for quick smoke tests).",
    )
    p.add_argument(
        "--force_restart",
        action="store_true",
        help="Delete existing meta and rebuild from scratch.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data_path = str(Path(args.data_path))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    build_shards(
        data_path=data_path,
        max_length=args.max_length,
        shard_size=args.shard_size,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        force_restart=args.force_restart,
    )


if __name__ == "__main__":
    main()
