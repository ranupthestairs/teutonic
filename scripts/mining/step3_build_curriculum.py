#!/usr/bin/env python3
"""Step 3: build the curriculum from scored samples."""
from __future__ import annotations

import argparse
from pathlib import Path

from challenger_step_lib import build_curriculum
from train_challenger import log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--scored", default="",
                    help="Scored sample JSONL; defaults to <work>/scored_samples.jsonl")
    ap.add_argument("--out-dir", default="",
                    help="Curriculum output dir; defaults to <work>/curriculum")
    ap.add_argument("--train-per-iter", type=int, default=20000,
                    help="Training sequences to keep after bucketing")
    ap.add_argument("--val-size", type=int, default=2000,
                    help="Validation sequences to keep after bucketing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    work = Path(args.work)
    scored = Path(args.scored) if args.scored else work / "scored_samples.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else work / "curriculum"

    build_curriculum(scored, out_dir, args.train_per_iter, args.val_size, args.seed)
    log.info("step3 complete: curriculum_dir=%s", out_dir)


if __name__ == "__main__":
    main()
