#!/usr/bin/env python3
"""Step 2: pull training shards and score sample sequences with the king."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download

from challenger_step_lib import read_json, score_samples, write_json
from train_challenger import download_shard, fetch_manifest, load_shard, log, sha256_dir


DEFAULT_KING_URL = (
    "https://huggingface.co/"
    "bluecolor/teutonic-q3-10b-5ek5koe5-10416140412-rn"
)
DEFAULT_KING_REVISION = "main"
MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.txt",
]


def repo_from_hf_link(model: str) -> str:
    model = model.strip()
    if not model:
        raise ValueError("model URL/repo cannot be empty")

    if "://" not in model:
        return model.removeprefix("models/").strip("/")

    parsed = urlparse(model)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"expected a huggingface.co model URL, got {model!r}")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if parts and parts[0] == "models":
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"could not parse Hugging Face model repo from {model!r}")
    return "/".join(parts[:2])


def has_transformers_model_files(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob(pattern) for pattern in ("*.safetensors", "*.bin"))


def king_repo_from_meta(meta: dict) -> str:
    dashboard_king = meta.get("dashboard_king") or {}
    return (
        meta.get("king_repo")
        or meta.get("model_repo")
        or meta.get("hf_repo")
        or dashboard_king.get("model_repo")
        or dashboard_king.get("hf_repo")
        or ""
    )


def king_revision_from_meta(meta: dict) -> str:
    dashboard_king = meta.get("dashboard_king") or {}
    return (
        meta.get("king_revision")
        or meta.get("revision")
        or dashboard_king.get("king_revision")
        or dashboard_king.get("king_digest")
        or dashboard_king.get("revision")
        or ""
    )


def candidate_king_meta_paths(work: Path) -> list[Path]:
    paths = [
        work / "king.json",
        work.parent / "work" / "king.json",
        Path("/root/teutonic-mining/work/king.json"),
    ]
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def resolve_king_dir(
    work: Path,
    king_dir_arg: str,
    repo_arg: str,
    model_url: str,
    revision_arg: str,
    hf_token: str,
    download_workers: int,
) -> tuple[Path, dict]:
    meta_path = work / "king.json"
    meta = {}
    if not king_dir_arg:
        for candidate in candidate_king_meta_paths(work):
            if candidate.exists():
                meta_path = candidate
                meta = read_json(candidate)
                break

    king_dir = Path(king_dir_arg) if king_dir_arg else Path(meta.get("king_dir", work / "king"))

    if has_transformers_model_files(king_dir):
        return king_dir, meta

    repo = repo_arg.strip() or king_repo_from_meta(meta) or repo_from_hf_link(model_url)
    revision = revision_arg.strip() or king_revision_from_meta(meta) or DEFAULT_KING_REVISION

    log.info(
        "king model dir missing/incomplete, downloading repo=%s revision=%s -> %s",
        repo,
        revision or "HEAD",
        king_dir,
    )
    king_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        revision=revision or None,
        local_dir=str(king_dir),
        allow_patterns=MODEL_ALLOW_PATTERNS,
        ignore_patterns="optimizer*",
        token=hf_token or None,
        max_workers=download_workers,
    )

    if not has_transformers_model_files(king_dir):
        raise FileNotFoundError(f"downloaded king model is still incomplete: {king_dir}")

    meta.update({
        "king_repo": repo,
        "king_revision": revision,
        "king_hash": sha256_dir(king_dir),
        "king_dir": str(king_dir),
        "dashboard_king": meta.get("dashboard_king") or {
            "model_repo": repo,
            "hf_repo": repo,
            "king_revision": revision,
            "king_digest": revision,
            "source": model_url,
        },
    })
    write_json(meta_path, meta)
    return king_dir, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--n-shards", type=int, default=2,
                    help="Number of training dataset shards to download")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first training shard")
    ap.add_argument("--eval-shard", type=int, default=10,
                    help="Reserved held-out shard index; must not overlap training shards")
    ap.add_argument("--n-score", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-url", default=DEFAULT_KING_URL,
                    help="Hugging Face model URL or repo to download if king is incomplete")
    ap.add_argument("--repo", default="",
                    help="Hugging Face repo id; overrides --model-url and metadata")
    ap.add_argument("--revision", default="",
                    help="Hugging Face revision; defaults to metadata or main")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                    help="Hugging Face token; defaults to HF_TOKEN")
    ap.add_argument("--download-workers", type=int, default=16,
                    help="Parallel workers if step2 must download a missing/incomplete king")
    ap.add_argument("--scored-out", default="",
                    help="Output scored JSONL; defaults to <work>/scored_samples.jsonl")
    ap.add_argument("--summary-out", default="",
                    help="Output summary JSON; defaults to <work>/score_summary.json")
    args = ap.parse_args()

    work = Path(args.work)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    scored_out = Path(args.scored_out) if args.scored_out else work / "scored_samples.jsonl"
    summary_out = Path(args.summary_out) if args.summary_out else work / "score_summary.json"

    king_dir, king_meta = resolve_king_dir(
        work,
        args.king_dir,
        args.repo,
        args.model_url,
        args.revision,
        args.hf_token,
        args.download_workers,
    )

    train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
    if args.eval_shard in train_shard_idxs:
        raise ValueError("eval_shard cannot overlap training shards")

    manifest = fetch_manifest(cache)
    shards = []
    shard_records = []
    for idx in train_shard_idxs:
        key = manifest["shards"][idx]["key"]
        path = cache / Path(key).name
        download_shard(key, path)
        arr, _ = load_shard(path)
        log.info("loaded training shard %d: %d sequences", idx, len(arr))
        shards.append(arr)
        shard_records.append({"idx": idx, "key": key, "path": str(path), "n_sequences": len(arr)})

    summary = score_samples(str(king_dir), shards, args.n_score, args.seed, args.device, scored_out)
    summary.update({
        "king_dir": str(king_dir),
        "king_repo": king_meta.get("king_repo"),
        "king_revision": king_meta.get("king_revision"),
        "king_hash": king_meta.get("king_hash"),
        "train_shards": shard_records,
        "eval_shard": args.eval_shard,
        "seed": args.seed,
    })
    write_json(summary_out, summary)
    log.info("step2 complete: scored=%s summary=%s", scored_out, summary_out)


if __name__ == "__main__":
    main()
