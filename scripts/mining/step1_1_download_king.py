#!/usr/bin/env python3
"""Step 1: download the current king directly from a Hippius Hub model link."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from challenger_step_lib import download_king_from_hippius, write_king_metadata
from train_challenger import log


DEFAULT_KING_URL = (
    "https://hub.hippius.com/models/"
    "mastertensor/teutonic-q3-10b-5ek5koe5-45621114678-rn"
)
DEFAULT_KING_REVISION = "5Ek5KoE5-45621114678-rn"


def repo_from_hub_link(model: str) -> str:
    """Accept a Hippius Hub model URL, registry ref, or raw namespace/name repo."""
    model = model.strip()
    if not model:
        raise ValueError("model URL/repo cannot be empty")

    if "://" not in model and not model.startswith("registry.hippius.com/"):
        return model.removeprefix("models/").strip("/")

    if model.startswith("registry.hippius.com/"):
        model = f"docker://{model}"

    parsed = urlparse(model)
    path = parsed.path.strip("/")
    if path.startswith("models/"):
        path = path[len("models/"):]
    if parsed.netloc == "registry.hippius.com":
        path = path.split(":", 1)[0]
    if path.count("/") < 1:
        raise ValueError(f"could not parse Hippius model repo from {model!r}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="Output model directory; defaults to <work>/king")
    ap.add_argument("--metadata-out", default="",
                    help="Output metadata JSON; defaults to <work>/king.json")
    ap.add_argument("--download-workers", type=int, default=1,
                    help="Parallel workers for Hippius model download")
    ap.add_argument("--model-url", default=DEFAULT_KING_URL,
                    help="Hippius Hub model URL, registry ref, or repo id")
    ap.add_argument("--repo", default="",
                    help="Hippius repo id; overrides --model-url")
    ap.add_argument("--revision", default=DEFAULT_KING_REVISION,
                    help="Hippius model version/revision tag")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    king_dir = Path(args.king_dir) if args.king_dir else work / "king"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "king.json"

    repo = args.repo.strip() or repo_from_hub_link(args.model_url)
    revision = args.revision.strip()
    king = {
        "model_repo": repo,
        "hf_repo": repo,
        "king_revision": revision,
        "king_digest": revision,
        "source": args.model_url,
    }
    repo, revision = download_king_from_hippius(king, king_dir, args.download_workers)
    write_king_metadata(metadata_out, king, king_dir, repo, revision)
    log.info("step1 complete: king_dir=%s metadata=%s", king_dir, metadata_out)


if __name__ == "__main__":
    main()
