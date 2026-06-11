#!/usr/bin/env python3
"""Train a Teutonic challenger and keep the merged model local.

This is a no-upload variant of train_challenger.py. It downloads the current
king from Hippius Hub, builds the training curriculum, trains/merges/evaluates
challengers, and writes the final verdict JSON. It intentionally has no upload
flag and never pushes the merged model to a remote Hub.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hippius_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_challenger import (
    fetch_king,
    fetch_manifest,
    download_shard,
    load_shard,
    log,
    paired_eval,
    run_lora_training,
    score_and_curate,
    sha256_dir,
)

HIPPIUS_MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.txt",
]


def parse_false_only(value: str) -> bool:
    if value.lower() not in {"0", "false", "no"}:
        raise argparse.ArgumentTypeError(
            "train_challenger_no_upload.py does not implement noise-only training"
        )
    return False


def download_king_from_hippius(king: dict, out_dir: Path, max_workers: int) -> tuple[str, str]:
    repo = king.get("model_repo") or king.get("hf_repo")
    if not repo:
        raise KeyError(f"dashboard king missing model_repo/hf_repo; keys={sorted(king.keys())}")
    revision = king.get("king_digest") or king.get("king_revision") or king.get("revision") or ""

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("downloading king from Hippius: repo=%s revision=%s -> %s",
             repo, (revision or "HEAD")[:19], out_dir)
    snapshot_download(
        repo_id=repo,
        revision=revision or None,
        local_dir=str(out_dir),
        allow_patterns=HIPPIUS_MODEL_ALLOW_PATTERNS,
        ignore_patterns="optimizer*",
        max_workers=max_workers,
    )
    return repo, revision


def merge_lora_local(base_model: str, adapter: Path, out: Path) -> Path:
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel

    resolved_base_model = str(Path(base_model).resolve())
    if resolved_base_model not in sys.path:
        sys.path.insert(0, resolved_base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        tok.save_pretrained(str(out))
    except Exception as exc:
        log.warning("tokenizer save skipped: %s", exc)

    # Preserve local Hippius snapshot config/code metadata without calling HF.
    for pattern in ("config.json", "generation_config.json", "*.py", "tokenizer*", "special_tokens*", "*.model"):
        for src in Path(base_model).glob(pattern):
            if src.is_file():
                shutil.copy(src, out / src.name)

    del base, merged
    torch.cuda.empty_cache()
    log.info("merged model saved to %s", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Working dir on this box")
    ap.add_argument("--bundle", default="/root/teutonic-mining/bundle",
                    help="Path to training_bundle directory")
    ap.add_argument("--n-shards", type=int, default=2,
                    help="Number of dataset shards to download for training")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first shard to use (other than eval shard)")
    ap.add_argument("--eval-shard", type=int, default=10,
                    help="Held-out shard index for offline paired eval")
    ap.add_argument("--n-eval", type=int, default=2000,
                    help="Sequences for offline paired eval (validator uses 20k)")
    ap.add_argument("--n-score", type=int, default=4000)
    ap.add_argument("--train-per-iter", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--max-iters", type=int, default=3,
                    help="Retry training with new seed if first attempt insufficient")
    ap.add_argument("--target-mu", type=float, default=0.05,
                    help="Stop training as soon as offline mu_hat exceeds this")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--noise-only", type=parse_false_only, default=False,
                    help="Compatibility no-op; only false/0/no is accepted")
    ap.add_argument("--download-workers", type=int, default=1,
                    help="Parallel workers for Hippius model download")
    ap.add_argument("--report-out", default="",
                    help="Write a final JSON verdict to this path")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. king
    king = fetch_king()
    king_dir = work / "king"
    king_repo, king_revision = download_king_from_hippius(
        king, king_dir, args.download_workers,
    )
    king_hash = sha256_dir(king_dir)
    log.info("king sha256[:16]=%s", king_hash[:16])

    # 2. dataset shards
    manifest = fetch_manifest(cache)
    train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
    if args.eval_shard in train_shard_idxs:
        raise ValueError("eval_shard cannot overlap training shards")
    shards = []
    for idx in train_shard_idxs:
        key = manifest["shards"][idx]["key"]
        path = cache / Path(key).name
        download_shard(key, path)
        arr, _ = load_shard(path)
        log.info("loaded shard %d: %d sequences", idx, len(arr))
        shards.append(arr)

    eval_key = manifest["shards"][args.eval_shard]["key"]
    eval_path = cache / Path(eval_key).name
    download_shard(eval_key, eval_path)
    eval_arr, _ = load_shard(eval_path)
    rng_eval = np.random.default_rng(0xE1A)
    eval_indices = rng_eval.choice(
        len(eval_arr), size=min(args.n_eval, len(eval_arr)), replace=False,
    ).tolist()
    log.info("held-out eval shard %d: %d sequences (sampling %d)",
             args.eval_shard, len(eval_arr), len(eval_indices))

    best = None
    history = []
    for it in range(args.max_iters):
        log.info("=" * 60)
        log.info("=== iteration %d/%d ===", it + 1, args.max_iters)
        log.info("=" * 60)
        seed = args.seed + 1000 * it

        # 3+4. score+curate
        iter_work = work / f"iter_{it:02d}"
        iter_work.mkdir(exist_ok=True)
        train_p, val_p = score_and_curate(
            str(king_dir), shards, args.n_score,
            args.train_per_iter, args.val_size, seed, "cuda:0", iter_work,
        )

        # 5. LoRA train
        out_dir = iter_work / "lora_out"
        adapter = run_lora_training(
            str(king_dir), train_p, val_p, out_dir, args.n_gpus, args,
            Path(args.bundle),
        )

        # 6. merge
        merged_dir = iter_work / "merged"
        merge_lora_local(str(king_dir), adapter, merged_dir)

        # 7. paired eval
        verdict = paired_eval(
            str(king_dir), str(merged_dir), eval_arr, eval_indices, "cuda:0",
        )
        verdict["iter"] = it
        verdict["seed"] = seed
        history.append(verdict)
        json.dump(verdict, open(iter_work / "verdict.json", "w"), indent=2)

        if best is None or verdict["mu_hat"] > best["mu_hat"]:
            best = {**verdict, "iter_dir": str(iter_work),
                    "merged_dir": str(merged_dir)}
        if verdict["mu_hat"] >= args.target_mu and verdict["accepted"]:
            log.info("target reached at iter %d", it)
            break

    final = {
        "king_repo": king_repo,
        "king_revision": king_revision,
        "king_hash": king_hash,
        "best": best,
        "history": history,
        "ts": time.time(),
    }
    if best:
        final["challenger_hash"] = sha256_dir(Path(best["merged_dir"]))

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("wrote verdict to %s", args.report_out)

    log.info("DONE - best mu_hat=%.6f accepted=%s merged_dir=%s",
             best["mu_hat"] if best else float("nan"),
             best["accepted"] if best else False,
             best["merged_dir"] if best else "")


if __name__ == "__main__":
    main()
