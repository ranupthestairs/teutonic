"""Shared helpers for stepwise Teutonic challenger training scripts."""
from __future__ import annotations

import json
import shutil
import ctypes
import inspect
import site
import sys
from pathlib import Path

import numpy as np
import torch
from hippius_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_challenger import (
    compute_per_seq_loss,
    log,
    sha256_dir,
)

HIPPIUS_MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.txt",
]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")


def jsonl_rows(path: str | Path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def preload_cuda_runtime() -> None:
    """Make Python-packaged CUDA runtimes visible to custom model extensions."""
    candidates = []
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        candidates.extend(base.glob("cuda_runtime/lib/libcudart.so*"))
    for cuda_runtime in candidates:
        try:
            ctypes.CDLL(str(cuda_runtime), mode=ctypes.RTLD_GLOBAL)
            log.info("preloaded CUDA runtime: %s", cuda_runtime)
            return
        except OSError:
            continue


def add_model_dir_to_pythonpath(model_dir: str | Path) -> None:
    """Support custom model files that import sibling files as top-level modules."""
    resolved = str(Path(model_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        log.info("added model dir to python path: %s", resolved)


def patch_transformers_masking_utils() -> None:
    """Bridge Quasar custom code to the installed Transformers mask helper."""
    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        return

    create_causal_mask = masking_utils.create_causal_mask
    if "cache_position" in inspect.signature(create_causal_mask).parameters:
        return

    def create_causal_mask_compat(*args, cache_position=None, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        original_get_mask_sizes = getattr(past_key_values, "get_mask_sizes", None)
        if cache_position is not None and original_get_mask_sizes is not None:
            def get_mask_sizes_compat(q_length_or_cache_position, layer_idx):
                if hasattr(q_length_or_cache_position, "shape"):
                    return original_get_mask_sizes(q_length_or_cache_position, layer_idx)
                return original_get_mask_sizes(cache_position, layer_idx)

            past_key_values.get_mask_sizes = get_mask_sizes_compat
        try:
            return create_causal_mask(*args, **kwargs)
        finally:
            if original_get_mask_sizes is not None:
                past_key_values.get_mask_sizes = original_get_mask_sizes

    masking_utils.create_causal_mask = create_causal_mask_compat
    log.info("patched transformers.masking_utils.create_causal_mask compatibility")


def download_king_from_hippius(king: dict, out_dir: Path, max_workers: int) -> tuple[str, str]:
    repo = king.get("model_repo") or king.get("hf_repo")
    if not repo:
        raise KeyError(f"dashboard king missing model_repo/hf_repo; keys={sorted(king.keys())}")
    revision = king.get("king_digest") or king.get("king_revision") or king.get("revision") or ""

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("downloading king from Hippius: repo=%s revision=%s -> %s",
             repo, (revision or "HEAD")[:19], out_dir)
    snapshot_download(
        repo_id=repo,
        revision=revision or None,
        local_dir=str(out_dir),
        allow_patterns=HIPPIUS_MODEL_ALLOW_PATTERNS,
        ignore_patterns="optimizer*",
        max_workers=max_workers,
    )
    return repo, revision


def write_king_metadata(path: Path, king: dict, king_dir: Path, repo: str, revision: str) -> None:
    write_json(path, {
        "king_repo": repo,
        "king_revision": revision,
        "king_hash": sha256_dir(king_dir),
        "king_dir": str(king_dir),
        "dashboard_king": king,
    })


def score_samples(
    king_dir: str,
    shards: list[np.ndarray],
    n_score: int,
    seed: int,
    device: str,
    out_path: Path,
) -> dict:
    rng = np.random.default_rng(seed)
    cands = []
    for s_idx, shard in enumerate(shards):
        if len(shard) == 0:
            continue
        n_take = max((n_score * 2) // max(len(shards), 1), 32)
        idxs = rng.choice(len(shard), size=min(n_take, len(shard)), replace=False)
        for j in idxs:
            cands.append((s_idx, int(j)))
    rng.shuffle(cands)

    log.info("scoring %d samples with king on %s", len(cands), device)
    preload_cuda_runtime()
    add_model_dir_to_pythonpath(king_dir)
    patch_transformers_masking_utils()
    model = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True, trust_remote_code=True,
    )
    model.eval()
    vocab_size = getattr(model.config, "vocab_size", None) or model.lm_head.out_features

    valid_cands = []
    invalid_count = 0
    for s_idx, j in cands:
        tokens = shards[s_idx][j]
        if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
            invalid_count += 1
            continue
        valid_cands.append((s_idx, j))
        if len(valid_cands) >= n_score:
            break
    cands = valid_cands
    if invalid_count:
        log.info("dropped %d sampled sequences with token ids outside vocab_size=%d",
                 invalid_count, vocab_size)
    if len(cands) < n_score:
        log.warning("only %d/%d sampled sequences fit vocab_size=%d",
                    len(cands), n_score, vocab_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    losses = []
    batch_size = 8
    with out_path.open("w") as f:
        for i in range(0, len(cands), batch_size):
            chunk = cands[i:i + batch_size]
            toks = [shards[s][j].tolist() for s, j in chunk]
            batch_losses = compute_per_seq_loss(model, toks, device)
            for (s_idx, j), tok, loss in zip(chunk, toks, batch_losses):
                arr = np.asarray(tok)
                unique_r = float(len(set(tok)) / len(tok))
                rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
                ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
                rep_ng = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
                losses.append(float(loss))
                f.write(json.dumps({
                    "shard": s_idx,
                    "idx": j,
                    "loss": float(loss),
                    "unique_r": unique_r,
                    "rep_r": rep_r,
                    "rep_ng4": rep_ng,
                    "tokens": tok,
                }) + "\n")
            if (i // batch_size) % 20 == 0:
                log.info("scored %d/%d samples", min(i + batch_size, len(cands)), len(cands))

    del model
    torch.cuda.empty_cache()

    summary = {
        "n_scored": len(losses),
        "loss_min": float(np.min(losses)) if losses else None,
        "loss_max": float(np.max(losses)) if losses else None,
        "loss_mean": float(np.mean(losses)) if losses else None,
        "scored_path": str(out_path),
    }
    log.info("wrote scored samples -> %s", out_path)
    return summary


def bucket_for_row(row: dict, p50: float, p85: float) -> str:
    if row["rep_r"] > 0.2 or row["rep_ng4"] > 0.5 or row["unique_r"] < 0.05:
        return "suspicious"
    if row["loss"] >= p85:
        return "hard"
    if row["loss"] >= p50 * 0.8:
        return "general"
    return "easy"


def build_curriculum(
    scored_path: Path,
    out_dir: Path,
    train_per_iter: int,
    val_size: int,
    seed: int,
) -> dict:
    rows = list(jsonl_rows(scored_path))
    if not rows:
        raise ValueError(f"no scored rows found in {scored_path}")

    losses = np.asarray([r["loss"] for r in rows])
    p50 = float(np.percentile(losses, 50))
    p85 = float(np.percentile(losses, 85))
    for row in rows:
        row["bucket"] = bucket_for_row(row, p50, p85)

    counts = {b: sum(1 for r in rows if r["bucket"] == b)
              for b in ("general", "hard", "easy", "suspicious")}
    clean = [r for r in rows if r["bucket"] != "suspicious"]

    rng = np.random.default_rng(seed + 1)
    rng.shuffle(clean)
    val_rows = clean[:val_size]
    val_keys = {(r["shard"], r["idx"]) for r in val_rows}
    pool = [r for r in clean if (r["shard"], r["idx"]) not in val_keys]

    general = [r for r in pool if r["bucket"] == "general"]
    hard = [r for r in pool if r["bucket"] == "hard"]
    easy = [r for r in pool if r["bucket"] == "easy"]
    n_general = int(train_per_iter * 0.6)
    n_hard = int(train_per_iter * 0.3)
    n_easy = train_per_iter - n_general - n_hard

    train_rows = []
    for src, n in ((general, n_general), (hard, n_hard), (easy, n_easy)):
        if not src:
            continue
        if n >= len(src):
            train_rows.extend(src)
        else:
            selected = rng.choice(len(src), size=n, replace=False)
            train_rows.extend(src[int(k)] for k in selected)
    rng.shuffle(train_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    bucketed_path = out_dir / "scored_bucketed.jsonl"

    with train_path.open("w") as f:
        for row in train_rows:
            f.write(json.dumps({"input_ids": row["tokens"]}) + "\n")
    with val_path.open("w") as f:
        for row in val_rows:
            f.write(json.dumps({"input_ids": row["tokens"]}) + "\n")
    with bucketed_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "counts": counts,
        "p50": p50,
        "p85": p85,
        "train": len(train_rows),
        "val": len(val_rows),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "bucketed_path": str(bucketed_path),
    }
    write_json(out_dir / "scoring.json", summary)
    log.info("wrote curriculum train=%d val=%d -> %s",
             len(train_rows), len(val_rows), out_dir)
    return summary


def merge_lora_local(base_model: str, adapter: Path, out: Path) -> Path:
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel

    preload_cuda_runtime()
    add_model_dir_to_pythonpath(base_model)
    patch_transformers_masking_utils()
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        tok.save_pretrained(str(out))
    except Exception as exc:
        log.warning("tokenizer save skipped: %s", exc)

    for pattern in ("config.json", "generation_config.json", "*.py", "tokenizer*", "special_tokens*", "*.model"):
        for src in Path(base_model).glob(pattern):
            if src.is_file():
                shutil.copy(src, out / src.name)

    del base, merged
    torch.cuda.empty_cache()
    log.info("merged model saved to %s", out)
    return out
