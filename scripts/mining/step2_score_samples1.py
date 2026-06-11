#!/usr/bin/env python3
"""Step 2: pull training shards and score sample sequences with the king."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
DEFAULT_DATASET_CONFIG = Path(__file__).with_name("datasets.txt")


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


def dataset_base_url(manifest_url: str) -> str:
    parsed = urlparse(manifest_url)
    marker = "/dataset/"
    if marker not in parsed.path:
        raise ValueError(f"manifest URL must contain /dataset/: {manifest_url!r}")
    base_path = parsed.path.split(marker, 1)[0].rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def dataset_name_from_manifest_url(manifest_url: str) -> str:
    parsed = urlparse(manifest_url)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        dataset_idx = parts.index("dataset")
    except ValueError as exc:
        raise ValueError(f"manifest URL must contain /dataset/: {manifest_url!r}") from exc
    if dataset_idx + 1 >= len(parts):
        raise ValueError(f"manifest URL is missing dataset name: {manifest_url!r}")
    name = parts[dataset_idx + 1]
    for suffix in ("-quasar-10b", "-qwen3-8b"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def read_dataset_config(path: Path) -> list[dict]:
    """Parse datasets.txt into weighted manifest entries.

    The file format is intentionally simple:
      - first section: manifest URLs
      - second section: "<dataset-name>: <ratio>" or "<dataset-name>: <N> shard(s)"

    Fixed shard counts are reserved first. Remaining --n-shards budget is split
    across ratio entries.
    """
    urls: list[str] = []
    ratios: dict[str, float] = {}
    fixed_counts: dict[str, int] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("http://", "https://")):
            urls.append(line)
            continue
        if ":" in line:
            name, value = line.split(":", 1)
            name = name.strip()
            value = value.strip().lower()
            if "shard" in value:
                fixed_counts[name] = int(value.split()[0])
            else:
                ratios[name] = float(value)

    entries = []
    for url in urls:
        name = dataset_name_from_manifest_url(url)
        if name not in ratios and name not in fixed_counts:
            log.info("ignoring dataset manifest without ratio/count: name=%s url=%s", name, url)
            continue
        entry = {
            "name": name,
            "manifest_url": url,
            "base_url": dataset_base_url(url),
        }
        if name in fixed_counts:
            entry["fixed_count"] = fixed_counts[name]
        if name in ratios:
            entry["ratio"] = ratios[name]
        entries.append(entry)

    requested_names = set(ratios) | set(fixed_counts)
    missing = sorted(requested_names - {entry["name"] for entry in entries})
    if missing:
        raise ValueError(f"dataset entries have no matching manifest URLs: {missing}")
    if not entries:
        raise ValueError(f"no datasets found in {path}")

    weighted_entries = [entry for entry in entries if "ratio" in entry]
    total = sum(entry["ratio"] for entry in weighted_entries)
    if weighted_entries and total <= 0:
        raise ValueError("dataset ratios must sum to a positive number")
    for entry in weighted_entries:
        entry["weight"] = entry["ratio"] / total
    return entries


def allocate_counts(total: int, weights: list[float]) -> list[int]:
    if total < 0:
        raise ValueError("total shard count cannot be negative")
    raw = [total * weight for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def allocate_dataset_counts(total: int, entries: list[dict]) -> list[int]:
    fixed_total = sum(entry.get("fixed_count", 0) for entry in entries)
    if fixed_total > total:
        raise ValueError(
            f"fixed dataset shard count ({fixed_total}) exceeds --n-shards ({total})"
        )

    counts = [entry.get("fixed_count", 0) for entry in entries]
    weighted_indices = [i for i, entry in enumerate(entries) if "weight" in entry]
    weighted_budget = total - fixed_total
    weighted_counts = allocate_counts(
        weighted_budget,
        [entries[i]["weight"] for i in weighted_indices],
    )
    for idx, count in zip(weighted_indices, weighted_counts):
        counts[idx] += count
    return counts


def fetch_manifest_url(entry: dict, cache: Path) -> dict:
    manifest_path = cache / entry["name"] / "manifest.json"
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s manifest from %s", entry["name"], entry["manifest_url"])
        subprocess.check_call(["curl", "-fsSL", "-o", str(manifest_path), entry["manifest_url"]])
    return json.loads(manifest_path.read_text())


def download_dataset_shard(entry: dict, shard_key: str, out: Path) -> Path:
    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cached: %s (%.1f GB)", out, out.stat().st_size / 1e9)
        return out
    url = f"{entry['base_url'].rstrip('/')}/{shard_key.lstrip('/')}"
    log.info("downloading %s:%s -> %s", entry["name"], shard_key, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out


def select_shard_indices(start: int, count: int, reserved_idx: int, total_available: int) -> list[int]:
    selected = []
    idx = start
    while len(selected) < count and idx < total_available:
        if idx != reserved_idx:
            selected.append(idx)
        idx += 1
    if len(selected) < count:
        raise ValueError(
            f"requested {count} shards from start={start}, but only selected "
            f"{len(selected)} before manifest ended at {total_available}"
        )
    return selected


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
    ap.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG),
                    help="datasets.txt with manifest URLs and dataset ratios")
    ap.add_argument("--single-manifest", action="store_true",
                    help="Use the legacy single manifest from train_challenger.fetch_manifest")
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

    shards = []
    shard_records = []
    # if args.single_manifest:
    train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
    if args.eval_shard in train_shard_idxs:
        raise ValueError("eval_shard cannot overlap training shards")

    manifest = fetch_manifest(cache)
    for idx in train_shard_idxs:
        key = manifest["shards"][idx]["key"]
        path = cache / Path(key).name
        download_shard(key, path)
        arr, _ = load_shard(path)
        log.info("loaded training shard %d: %d sequences", idx, len(arr))
        shards.append(arr)
        shard_records.append({
            "dataset": "legacy",
            "idx": idx,
            "key": key,
            "path": str(path),
            "n_sequences": len(arr),
        })
    # else:
    #     dataset_entries = read_dataset_config(Path(args.dataset_config))
    #     counts = allocate_dataset_counts(args.n_shards, dataset_entries)
    #     log.info(
    #         "dataset shard mix: %s",
    #         ", ".join(
    #             f"{entry['name']}={count} "
    #             f"({entry.get('ratio', str(entry.get('fixed_count')) + ' shard')})"
    #             for entry, count in zip(dataset_entries, counts)
    #         ),
    #     )
    #     for entry, count in zip(dataset_entries, counts):
    #         if count == 0:
    #             continue
    #         manifest = fetch_manifest_url(entry, cache)
    #         selected_indices = select_shard_indices(
    #             args.shard_start,
    #             count,
    #             args.eval_shard,
    #             len(manifest["shards"]),
    #         )
    #         for idx in selected_indices:
    #             key = manifest["shards"][idx]["key"]
    #             path = cache / entry["name"] / Path(key).name
    #             download_dataset_shard(entry, key, path)
    #             arr, _ = load_shard(path)
    #             log.info(
    #                 "loaded %s training shard %d: %d sequences",
    #                 entry["name"],
    #                 idx,
    #                 len(arr),
    #             )
    #             shards.append(arr)
    #             shard_records.append({
    #                 "dataset": entry["name"],
    #                 "ratio": entry.get("ratio"),
    #                 "fixed_count": entry.get("fixed_count"),
    #                 "idx": idx,
    #                 "key": key,
    #                 "path": str(path),
    #                 "n_sequences": len(arr),
    #             })

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
