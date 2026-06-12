#!/usr/bin/env python3
"""Step 6: offline paired eval of merged challenger vs king and write verdict JSON."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from challenger_step_lib import paired_eval_datasets, read_json, write_json
from train_challenger import (
    DEFAULT_DATASETS,
    allocate_weighted_counts,
    download_shard,
    fetch_manifest_url,
    load_shard,
    log,
    sha256_dir,
)


def load_dataset_specs(datasets_config: str) -> list[dict]:
    if not datasets_config:
        return [dict(spec) for spec in DEFAULT_DATASETS]
    path = Path(datasets_config)
    data = json.loads(path.read_text()) if path.exists() else json.loads(datasets_config)
    if isinstance(data, dict):
        data = data.get("datasets", data.get("items", []))
    if not isinstance(data, list) or not data:
        raise ValueError("datasets config must be a non-empty list")
    return data


def select_shard_indices(
    n_shards: int,
    n_manifest_shards: int,
    seed: int,
    random_shards: bool,
    shard_start: int,
) -> list[int]:
    if n_manifest_shards <= 0:
        raise ValueError("manifest has no shards")
    if n_shards > n_manifest_shards:
        raise ValueError(
            f"requested {n_shards} shards, but manifest only has {n_manifest_shards}"
        )
    if random_shards:
        return sorted(random.Random(seed).sample(range(n_manifest_shards), n_shards))
    end = shard_start + n_shards
    if end > n_manifest_shards:
        raise ValueError(
            f"requested shard range [{shard_start}, {end}) exceeds manifest size "
            f"{n_manifest_shards}"
        )
    return list(range(shard_start, end))


def load_eval_sets(
    work: Path,
    datasets: list[dict],
    sample_counts: dict[str, int],
    n_shards_per_dataset: int,
    seed: int,
    random_shards: bool,
    shard_start: int,
) -> list[dict]:
    eval_sets = []
    for spec_idx, spec in enumerate(datasets):
        name = spec["name"]
        manifest_url = spec["manifest_url"]
        weight = float(spec["weight"])
        target_samples = int(sample_counts[name])
        dataset_cache = work / "cache" / "datasets" / name
        manifest = fetch_manifest_url(dataset_cache, manifest_url)
        shard_indices = select_shard_indices(
            n_shards_per_dataset,
            len(manifest["shards"]),
            seed + spec_idx,
            random_shards,
            shard_start,
        )
        log.info(
            "eval dataset %s: weight=%.2f target_samples=%d shards=%s",
            name,
            weight,
            target_samples,
            shard_indices,
        )

        shard = None
        shard_key = ""
        for manifest_shard_idx in shard_indices:
            shard_info = manifest["shards"][manifest_shard_idx]
            shard_key = shard_info["key"]
            path = dataset_cache / "shards" / Path(shard_key).name
            download_shard(shard_key, path, manifest=manifest)
            shard, _ = load_shard(path)
            if len(shard) >= target_samples:
                break
            log.warning(
                "dataset %s shard %d only has %d sequences; need %d",
                name,
                manifest_shard_idx,
                len(shard),
                target_samples,
            )

        if shard is None or len(shard) == 0:
            raise ValueError(f"no eval shard loaded for dataset {name}")

        rng_eval = np.random.default_rng(seed + 0xE1A + spec_idx)
        n_take = min(target_samples, len(shard))
        indices = rng_eval.choice(len(shard), size=n_take, replace=False).tolist()
        eval_sets.append({
            "dataset": name,
            "weight": weight,
            "target_samples": target_samples,
            "manifest_url": manifest_url,
            "shard_key": shard_key,
            "shard": shard,
            "indices": indices,
        })
        log.info(
            "eval dataset %s: sampled %d/%d sequences from %s",
            name,
            len(indices),
            len(shard),
            shard_key,
        )
    return eval_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--merged-dir", default="",
                    help="Merged model dir; defaults to merged_dir in <work>/merged.json")
    ap.add_argument("--datasets-config", default="",
                    help="Optional JSON file/list overriding DEFAULT_DATASETS")
    ap.add_argument("--n-shards-per-dataset", type=int, default=1,
                    help="Number of shards to download per dataset manifest")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first shard when not using --random-shards")
    ap.add_argument("--random-shards", action="store_true",
                    help="Randomly sample shards per dataset with --seed")
    ap.add_argument("--n-eval", type=int, default=2000,
                    help="Total sequences for offline paired eval across datasets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--verdict-out", default="",
                    help="Verdict JSON; defaults to <work>/verdict.json")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if args.king_dir:
        king_dir = Path(args.king_dir)
        king_meta = {}
    else:
        king_meta_path = work / "king.json"
        king_meta = read_json(king_meta_path) if king_meta_path.exists() else {}
        king_dir = Path(king_meta.get("king_dir", work / "king"))

    if args.merged_dir:
        merged_dir = Path(args.merged_dir)
        merged_meta = {}
    else:
        merged_meta_path = work / "merged.json"
        merged_meta = read_json(merged_meta_path) if merged_meta_path.exists() else {}
        merged_dir = Path(merged_meta.get("merged_dir", work / "merged"))

    verdict_out = Path(args.verdict_out) if args.verdict_out else work / "verdict.json"

    datasets = load_dataset_specs(args.datasets_config)
    weights = [float(spec["weight"]) for spec in datasets]
    counts = allocate_weighted_counts(args.n_eval, weights)
    sample_counts = {
        spec["name"]: count
        for spec, count in zip(datasets, counts)
    }
    log.info("eval allocation across datasets: %s", sample_counts)

    eval_sets = load_eval_sets(
        work,
        datasets,
        sample_counts,
        args.n_shards_per_dataset,
        args.seed,
        args.random_shards,
        args.shard_start,
    )

    verdict = paired_eval_datasets(
        str(king_dir),
        str(merged_dir),
        eval_sets,
        args.device,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
    )
    final = {
        "king_repo": king_meta.get("king_repo"),
        "king_revision": king_meta.get("king_revision"),
        "king_hash": king_meta.get("king_hash") or sha256_dir(king_dir),
        "challenger_hash": merged_meta.get("challenger_hash") or sha256_dir(merged_dir),
        "king_dir": str(king_dir),
        "merged_dir": str(merged_dir),
        "datasets_config": datasets,
        "sample_counts": sample_counts,
        "n_eval_requested": args.n_eval,
        "seed": args.seed,
        "verdict": verdict,
        "ts": time.time(),
    }
    write_json(verdict_out, final)
    log.info("step6 complete: verdict=%s", verdict_out)


if __name__ == "__main__":
    main()
