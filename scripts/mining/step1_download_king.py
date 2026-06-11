#!/usr/bin/env python3
"""Step 1: discover the current king and download it from Hippius Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

from challenger_step_lib import download_king_from_hippius, write_king_metadata
from train_challenger import fetch_king, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="Output model directory; defaults to <work>/king")
    ap.add_argument("--metadata-out", default="",
                    help="Output metadata JSON; defaults to <work>/king.json")
    ap.add_argument("--download-workers", type=int, default=1,
                    help="Parallel workers for Hippius model download")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    king_dir = Path(args.king_dir) if args.king_dir else work / "king"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "king.json"

    king = fetch_king()
    repo, revision = download_king_from_hippius(king, king_dir, args.download_workers)
    write_king_metadata(metadata_out, king, king_dir, repo, revision)
    log.info("step1 complete: king_dir=%s metadata=%s", king_dir, metadata_out)


if __name__ == "__main__":
    main()