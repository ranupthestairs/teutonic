#!/usr/bin/env python3
"""Offline paired eval of a LoRA checkpoint vs king without merging first."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from challenger_step_lib import paired_eval_lora_datasets, read_json, write_json
from step6_eval_verdict import load_dataset_specs, load_eval_sets
from train_challenger import allocate_weighted_counts, log, sha256_dir


def resolve_adapter_dir(work: Path, adapter_arg: str, checkpoint_arg: str) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)
    if adapter_arg:
        return Path(adapter_arg)

    adapter_meta_path = work / "adapter.json"
    if adapter_meta_path.exists():
        adapter_meta = read_json(adapter_meta_path)
        adapter_dir = adapter_meta.get("adapter_dir")
        if adapter_dir:
            return Path(adapter_dir)

    return work / "lora_out" / "best_adapter"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--checkpoint-dir", default="",
                    help="Specific LoRA checkpoint/adapter dir to evaluate")
    ap.add_argument("--adapter-dir", default="",
                    help="LoRA adapter dir; defaults to adapter_dir in <work>/adapter.json or <work>/lora_out/best_adapter")
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
                    help="Verdict JSON; defaults to <work>/lora_verdict.json")
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

    adapter_dir = resolve_adapter_dir(work, args.adapter_dir, args.checkpoint_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter/checkpoint dir not found: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{adapter_dir} does not look like a PEFT adapter dir; missing adapter_config.json"
        )

    verdict_out = Path(args.verdict_out) if args.verdict_out else work / "lora_verdict.json"

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

    verdict = paired_eval_lora_datasets(
        str(king_dir),
        str(adapter_dir),
        eval_sets,
        args.device,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
    )
    final = {
        "king_repo": king_meta.get("king_repo"),
        "king_revision": king_meta.get("king_revision"),
        "king_hash": king_meta.get("king_hash") or sha256_dir(king_dir),
        "king_dir": str(king_dir),
        "adapter_dir": str(adapter_dir),
        "adapter_hash": sha256_dir(adapter_dir),
        "datasets_config": datasets,
        "sample_counts": sample_counts,
        "n_eval_requested": args.n_eval,
        "seed": args.seed,
        "verdict": verdict,
        "ts": time.time(),
    }
    write_json(verdict_out, final)
    log.info("step6 lora complete: verdict=%s", verdict_out)


if __name__ == "__main__":
    main()
