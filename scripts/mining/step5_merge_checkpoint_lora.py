#!/usr/bin/env python3
"""Step 5b: merge a specific LoRA checkpoint into the king model."""
from __future__ import annotations

import argparse
from pathlib import Path

from challenger_step_lib import merge_lora_local, read_json, write_json
from train_challenger import log, sha256_dir


def resolve_checkpoint_dir(work: Path, checkpoint_arg: str, adapter_arg: str) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)
    if adapter_arg:
        return Path(adapter_arg)
    raise ValueError("one of --checkpoint-dir or --adapter-dir is required")


def default_merged_dir(work: Path, adapter_dir: Path) -> Path:
    name = adapter_dir.name or "adapter"
    return work / f"merged-{name}"


def default_metadata_out(work: Path, adapter_dir: Path) -> Path:
    name = adapter_dir.name or "adapter"
    return work / f"merged-{name}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--checkpoint-dir", default="",
                    help="Specific LoRA checkpoint dir to merge (e.g. lora_out/checkpoint-1600)")
    ap.add_argument("--adapter-dir", default="",
                    help="Alias for --checkpoint-dir")
    ap.add_argument("--merged-dir", default="",
                    help="Merged model output dir; defaults to <work>/merged-<checkpoint-name>")
    ap.add_argument("--max-shard-size", default="4.3GB",
                    help="Split model.safetensors into shards of at most this size "
                         "(Transformers format, e.g. 4.3GB). Use empty string for a single file.")
    ap.add_argument("--metadata-out", default="",
                    help="Merge metadata JSON; defaults to <work>/merged-<checkpoint-name>.json")
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

    adapter_dir = resolve_checkpoint_dir(work, args.checkpoint_dir, args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint dir not found: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{adapter_dir} does not look like a PEFT adapter dir; missing adapter_config.json"
        )

    merged_dir = Path(args.merged_dir) if args.merged_dir else default_merged_dir(work, adapter_dir)
    metadata_out = Path(args.metadata_out) if args.metadata_out else default_metadata_out(work, adapter_dir)

    merge_lora_local(
        str(king_dir),
        adapter_dir,
        merged_dir,
        max_shard_size=args.max_shard_size,
    )
    write_json(metadata_out, {
        "king_dir": str(king_dir),
        "king_repo": king_meta.get("king_repo"),
        "king_revision": king_meta.get("king_revision"),
        "king_hash": king_meta.get("king_hash") or sha256_dir(king_dir),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "challenger_hash": sha256_dir(merged_dir),
    })
    log.info("step5 checkpoint merge complete: merged=%s metadata=%s", merged_dir, metadata_out)


if __name__ == "__main__":
    main()
