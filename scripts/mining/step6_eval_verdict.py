#!/usr/bin/env python3
"""Step 6: offline paired eval of merged challenger vs king and write verdict JSON."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from challenger_step_lib import read_json, write_json
from train_challenger import (
    download_shard,
    fetch_manifest,
    load_shard,
    log,
    paired_eval,
    sha256_dir,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="/workspace/teutonic-mining/work/king",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--merged-dir", default="",
                    help="Merged model dir; defaults to merged_dir in <work>/merged.json")
    ap.add_argument("--eval-shard", type=int, default=10,
                    help="Held-out shard index for offline paired eval")
    ap.add_argument("--n-eval", type=int, default=2000,
                    help="Sequences for offline paired eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--verdict-out", default="",
                    help="Verdict JSON; defaults to <work>/verdict.json")
    args = ap.parse_args()

    work = Path(args.work)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

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

    manifest = fetch_manifest(cache)
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

    verdict = paired_eval(
        str(king_dir),
        str(merged_dir),
        eval_arr,
        eval_indices,
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
        "eval_shard": args.eval_shard,
        "n_eval_requested": args.n_eval,
        "verdict": verdict,
        "ts": time.time(),
    }
    write_json(verdict_out, final)
    log.info("step6 complete: verdict=%s", verdict_out)


if __name__ == "__main__":
    main()
