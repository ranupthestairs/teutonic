#!/usr/bin/env python3
"""Step 5: merge the trained LoRA adapter into the base king weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from challenger_step_lib import merge_lora_local, read_json, write_json
from train_challenger import log, sha256_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--adapter-dir", default="",
                    help="Adapter dir; defaults to adapter_dir in <work>/adapter.json")
    ap.add_argument("--merged-dir", default="",
                    help="Merged model output dir; defaults to <work>/merged")
    ap.add_argument("--metadata-out", default="",
                    help="Merge metadata JSON; defaults to <work>/merged.json")
    args = ap.parse_args()

    work = Path(args.work)
    if args.king_dir:
        king_dir = Path(args.king_dir)
    else:
        king_meta = work / "king.json"
        king_dir = Path(read_json(king_meta)["king_dir"]) if king_meta.exists() else work / "king"

    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
    else:
        adapter_dir = Path(read_json(work / "adapter.json")["adapter_dir"])

    merged_dir = Path(args.merged_dir) if args.merged_dir else work / "merged"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "merged.json"

    merge_lora_local(str(king_dir), adapter_dir, merged_dir)
    write_json(metadata_out, {
        "king_dir": str(king_dir),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "challenger_hash": sha256_dir(merged_dir),
    })
    log.info("step5 complete: merged=%s metadata=%s", merged_dir, metadata_out)


if __name__ == "__main__":
    main()
