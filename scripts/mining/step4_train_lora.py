#!/usr/bin/env python3
"""Step 4: train a LoRA adapter from curriculum JSONL files."""
from __future__ import annotations

import argparse
from pathlib import Path

from challenger_step_lib import read_json, write_json
from train_challenger import log, run_lora_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="/root/teutonic-mining/work/king",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--bundle", default="/root/teutonic-mining/bundle",
                    help="Path to training_bundle directory")
    ap.add_argument("--train-data", default="",
                    help="Train JSONL; defaults to <work>/curriculum/train.jsonl")
    ap.add_argument("--val-data", default="",
                    help="Validation JSONL; defaults to <work>/curriculum/val.jsonl")
    ap.add_argument("--output-dir", default="",
                    help="LoRA output dir; defaults to <work>/lora_out")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--metadata-out", default="",
                    help="Adapter metadata JSON; defaults to <work>/adapter.json")
    args = ap.parse_args()

    work = Path(args.work)
    if args.king_dir:
        king_dir = Path(args.king_dir)
    else:
        meta_path = work / "king.json"
        king_dir = Path(read_json(meta_path)["king_dir"]) if meta_path.exists() else work / "king"
    train_data = Path(args.train_data) if args.train_data else work / "curriculum" / "train.jsonl"
    val_data = Path(args.val_data) if args.val_data else work / "curriculum" / "val.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else work / "lora_out"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "adapter.json"

    adapter = run_lora_training(
        str(king_dir), train_data, val_data, output_dir, args.n_gpus, args, Path(args.bundle),
    )
    write_json(metadata_out, {
        "king_dir": str(king_dir),
        "train_data": str(train_data),
        "val_data": str(val_data),
        "lora_output_dir": str(output_dir),
        "adapter_dir": str(adapter),
    })
    log.info("step4 complete: adapter=%s metadata=%s", adapter, metadata_out)


if __name__ == "__main__":
    main()
