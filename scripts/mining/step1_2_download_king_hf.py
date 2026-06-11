#!/usr/bin/env python3
"""Step 1: download the current king directly from a Hugging Face model link."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download

from challenger_step_lib import write_king_metadata
from train_challenger import log


DEFAULT_KING_URL = (
    "https://huggingface.co/"
    "bluecolor/teutonic-q3-10b-5ek5koe5-10416140412-rn"
)
DEFAULT_KING_REVISION = "main"
HF_MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.txt",
]


def repo_from_hf_link(model: str) -> str:
    """Accept a Hugging Face model URL or raw namespace/name repo."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="Output model directory; defaults to <work>/king")
    ap.add_argument("--metadata-out", default="",
                    help="Output metadata JSON; defaults to <work>/king.json")
    ap.add_argument("--download-workers", type=int, default=16,
                    help="Parallel workers for Hugging Face model download")
    ap.add_argument("--model-url", default=DEFAULT_KING_URL,
                    help="Hugging Face model URL or repo id")
    ap.add_argument("--repo", default="",
                    help="Hugging Face repo id; overrides --model-url")
    ap.add_argument("--revision", default=DEFAULT_KING_REVISION,
                    help="Hugging Face revision, branch, tag, or commit")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                    help="Hugging Face token; defaults to HF_TOKEN")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    king_dir = Path(args.king_dir) if args.king_dir else work / "king"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "king.json"

    repo = args.repo.strip() or repo_from_hf_link(args.model_url)
    revision = args.revision.strip()

    if king_dir.exists():
        shutil.rmtree(king_dir)
    king_dir.mkdir(parents=True, exist_ok=True)

    log.info("downloading king from Hugging Face: repo=%s revision=%s -> %s",
             repo, revision or "HEAD", king_dir)
    snapshot_download(
        repo_id=repo,
        revision=revision or None,
        local_dir=str(king_dir),
        allow_patterns=HF_MODEL_ALLOW_PATTERNS,
        ignore_patterns="optimizer*",
        token=args.hf_token or None,
        max_workers=args.download_workers,
    )

    king = {
        "model_repo": repo,
        "hf_repo": repo,
        "king_revision": revision,
        "king_digest": revision,
        "source": args.model_url,
    }
    write_king_metadata(metadata_out, king, king_dir, repo, revision)
    log.info("step1 complete: king_dir=%s metadata=%s", king_dir, metadata_out)


if __name__ == "__main__":
    main()
