#!/usr/bin/env python3
import argparse
import ctypes
import inspect
import json
import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def preload_cuda_runtime() -> None:
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        for cuda_runtime in base.glob("cuda_runtime/lib/libcudart.so*"):
            try:
                ctypes.CDLL(str(cuda_runtime), mode=ctypes.RTLD_GLOBAL)
                print(f"preloaded CUDA runtime: {cuda_runtime}", flush=True)
                return
            except OSError:
                continue


def add_model_dir_to_pythonpath(model_dir: str) -> None:
    resolved = str(Path(model_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        print(f"added model dir to python path: {resolved}", flush=True)


def patch_transformers_masking_utils() -> None:
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
    print("patched transformers.masking_utils.create_causal_mask compatibility", flush=True)


def load_tokenizer_optional(base_model: str):
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                base_model,
                use_fast=use_fast,
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"tokenizer load failed use_fast={use_fast}: {exc}", flush=True)
    print("continuing without tokenizer; token-id training does not require one", flush=True)
    return None


class TokenIdsDataset(Dataset):
    def __init__(self, path, seq_len):
        self.rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    ids = obj["input_ids"][:seq_len]
                    if len(ids) < seq_len:
                        continue
                    self.rows.append(ids)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = self.rows[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


@dataclass
class Collator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", type=str, default=None,
                    help="comma-separated module name suffixes; defaults to a Quasar-aware set")
    args = ap.parse_args()

    preload_cuda_runtime()
    add_model_dir_to_pythonpath(args.base_model)
    patch_transformers_masking_utils()

    tokenizer = load_tokenizer_optional(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Quasar module names: attn uses q_proj/k_proj/v_proj/o_proj
    # like Qwen3, but FFN paths split into the dense SwiGLU (ffn.gate / ffn.up
    # / ffn.down) and BigMac MoE (shared_experts.{i}.{gate,up,down},
    # w_down_proj, w_up_proj). LoRA on `experts_w12` / `experts_w3` is not
    # supported — they are nn.Parameter blocks, not Linear layers. Override
    # via --lora-target-modules if you want a custom set.
    target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up", "down",
        "w_down_proj", "w_up_proj",
    ]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = TokenIdsDataset(args.train_data, args.seq_len)
    val_ds = TokenIdsDataset(args.val_data, args.seq_len)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_adapter"))
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(args.output_dir, "best_adapter"))


if __name__ == "__main__":
    main()
