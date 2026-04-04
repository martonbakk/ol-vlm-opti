"""Fine-tune Qwen VL model with nsys-compatible cudaProfilerStart/Stop around training."""

import argparse
import types
from pathlib import Path
from typing import Any

import torch
from torch.cuda import cudart, check_error
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
from src.data.data import QwenDataset

from src.token_pruner.sparsity import apply_qwen3_sparsity
from src.vlm_lora import VramTrainConfig, apply_qlora_peft, build_bitsandbytes_config
from src.vlm_opt import VlmOptConfig, apply_vlm_optimizations
from src.vlm_opt.config import MergerFusedBackend
from src.vlm_opt.io import safe_print


def _patch_module_zero_grad_set_to_none(module: torch.nn.Module) -> None:
    """Force ``zero_grad(set_to_none=True)`` to reduce memset-style clears (nsys)."""

    if getattr(module, "_ol_zero_grad_patched", False):
        return

    def zero_grad_always_none(self: torch.nn.Module, set_to_none: bool = True) -> None:
        torch.nn.Module.zero_grad(self, set_to_none=True)

    module.zero_grad = types.MethodType(zero_grad_always_none, module)  # type: ignore[method-assign]
    setattr(module, "_ol_zero_grad_patched", True)


class _ZeroGradSetToNoneCallback(TrainerCallback):
    """Patch model (and wrapped replica) after Trainer wiring — covers single-GPU + DDP-style wraps."""

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        seen: set[int] = set()
        for m in (self._trainer.model_wrapped, self._trainer.model):
            if m is None:
                continue
            mid = id(m)
            if mid in seen:
                continue
            seen.add(mid)
            if isinstance(m, torch.nn.Module):
                _patch_module_zero_grad_set_to_none(m)


def _collate_build_labels_from_batch(out: dict[str, Any]) -> None:
    """One clone per batch for labels (not per sample) — less D2D / VRAM churn than per-row clone."""
    input_ids = out["input_ids"]
    attention_mask = out["attention_mask"]
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    out["labels"] = labels


def _collate_fn(batch: list[dict[str, Any]], tokenizer: Any, pad_token_id: int = 0) -> dict[str, Any]:
    """Pad variable-length sequences in batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len
        input_ids.append(
            torch.cat(
                [
                    x["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=x["input_ids"].dtype),
                ]
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    x["attention_mask"],
                    torch.zeros(pad_len, dtype=x["attention_mask"].dtype),
                ]
            )
        )
        lb = x.get("labels")
        if lb is not None:
            labels.append(torch.cat([lb, torch.full((pad_len,), -100, dtype=lb.dtype)]))
    out = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }
    if labels:
        out["labels"] = torch.stack(labels)
    else:
        _collate_build_labels_from_batch(out)
    # Vision inputs: Qwen VL uses concat for variable-resolution batching
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        # Dataset uses squeeze(0), so single-sample grid_thw is (3,) not (1,3). Model expects (num_images, 3).
        grids = []
        for x in batch:
            g = x["image_grid_thw"]
            if g.dim() == 1:
                g = g.unsqueeze(0)
            grids.append(g)
        out["image_grid_thw"] = torch.cat(grids, dim=0)
    # Single contiguous layout before .to(device) can avoid extra D2D in some cudnn/cublas paths
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor) and not v.is_contiguous():
            out[k] = v.contiguous()
    return out


class _FinetuneCollator:
    """Picklable collator for ``DataLoader(num_workers>0)`` (Windows uses spawn; nested functions are not picklable)."""

    __slots__ = ("pad_token_id", "tokenizer")

    def __init__(self, tokenizer: Any, pad_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return _collate_fn(batch, self.tokenizer, self.pad_token_id)


def run_finetune(
    model: Any,
    processor: Any,
    dataset: Any,
    output_dir: str,
    epochs: float = 1.0,
    batch_size: int = 2,
    *,
    bf16: bool | None = None,
    optim: str | None = None,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    qlora: bool = False,
    dataloader_num_workers: int = 4,
    gradient_accumulation_steps: int = 1,
) -> None:
    """Run fine-tuning with model, processor and dataset already loaded (e.g. from notebook).

    ``bf16=None`` → enable bfloat16 when CUDA supports it (narrower activations = less memory traffic).
    ``optim=None`` → use fused AdamW on CUDA when available (fewer small GPU ops).
    ``gradient_checkpointing=True`` → trade compute for lower activation VRAM (strongest lever besides batch size).
    ``qlora=True`` → model is expected to be a PEFT QLoRA wrapper; logging only (saves adapter weights).
    ``dataloader_num_workers`` → background dataset workers (0 = main process only). Raise if the GPU waits on data;
    lower if CPU stays saturated and training stutters.
    ``gradient_accumulation_steps`` → effective batch = batch_size * this (does not lower peak VRAM per forward; use smaller images / sparsity for OOM).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collate = _FinetuneCollator(processor.tokenizer, pad_id)

    use_bf16 = (
        bf16
        if bf16 is not None
        else (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    )
    optim_name = optim
    if optim_name is None and torch.cuda.is_available():
        optim_name = "adamw_torch_fused"
    if optim_name is None:
        optim_name = "adamw_torch"

    dataset_len = len(dataset)
    if dataset_len >= 2:
        train_size = max(1, int(0.9 * dataset_len))
        eval_size = dataset_len - train_size
        if eval_size == 0:
            train_size = dataset_len - 1
            eval_size = 1
        train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        eval_dataset = torch.utils.data.Subset(
            dataset, range(train_size, train_size + eval_size)
        )
        use_eval = True
    else:
        print("Dataset has fewer than 2 samples; skipping eval split and early stopping.")
        train_dataset = dataset
        eval_dataset = None
        use_eval = False

    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "remove_unused_columns": False,
        "logging_steps": 5,
        "bf16": use_bf16,
        "fp16": False,
        "optim": optim_name,
        "seed": seed,
        "data_seed": seed,
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_persistent_workers": dataloader_num_workers > 0,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }

    safe_print(
        "[finetune] Training config - "
        f"bf16={use_bf16}, batch_size={batch_size}, "
        f"grad_accum={gradient_accumulation_steps}, "
        f"dataloader_num_workers={dataloader_num_workers}, "
        f"gradient_checkpointing={gradient_checkpointing}, qlora={qlora}, optim={optim_name!r}"
    )
    if use_eval:
        training_kwargs.update(
            {
                "eval_steps": 500,
                "save_steps": 500,
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "loss",
                "greater_is_better": False,
                "save_strategy": "steps",
            }
        )
    else:
        training_kwargs["save_strategy"] = "no"

    try:
        if use_eval:
            training_args = TrainingArguments(
                eval_strategy="steps",
                **training_kwargs,
            )
        else:
            training_args = TrainingArguments(**training_kwargs)
    except TypeError:
        if use_eval:
            training_args = TrainingArguments(
                evaluation_strategy="steps",
                **training_kwargs,
            )
        else:
            training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    trainer.add_callback(_ZeroGradSetToNoneCallback(trainer))
    if use_eval:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    print(
        "Starting training (cudaProfilerStart/Stop for nsys --capture-range=cudaProfilerApi)..."
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStart())
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStop())
    print("Saving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", default="test[:1%]")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--vision-max-pixels",
        type=int,
        default=262144,
        help="Processor max image pixels (H*W). Primary VRAM lever if OOM: try 131072, 65536, or 50176.",
    )
    parser.add_argument(
        "--vision-min-pixels",
        type=int,
        default=50176,
        help="Processor min image pixels (H*W). Must stay below or equal to --vision-max-pixels.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=768,
        help="Resize longest image side on CPU (pixels). Lower with --vision-max-pixels if CUDA OOM (e.g. 512 or 384).",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 even if the GPU supports it.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer name for TrainingArguments (default: adamw_torch_fused on CUDA else adamw_torch).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset shuffling and Trainer sampling.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        metavar="N",
        help="DataLoader worker processes (0 = load and collate in the main process only). "
        "Increase if the GPU often waits on batches; decrease if CPU usage stays at 100%% and steps stutter. "
        "Windows: use 0 if multiprocessing errors appear.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        metavar="N",
        help="Trainer gradient accumulation (effective batch = batch_size * N). Does not reduce peak VRAM per step; "
        "use smaller --vision-max-pixels / --image-max-side or --token-sparsity for OOM.",
    )
    parser.add_argument(
        "--vlm-opt",
        action="store_true",
        help="Enable src/vlm_opt: fused vision PatchMerger + optional Liger / full-model compile.",
    )
    parser.add_argument(
        "--vlm-opt-merger-backend",
        type=str,
        default="auto",
        choices=[b.value for b in MergerFusedBackend],
        help="Fused Linear+GELU backend: auto|pytorch|triton|compile_wrap (Windows: pytorch or compile_wrap recommended).",
    )
    parser.add_argument(
        "--vlm-opt-liger",
        action="store_true",
        help="Try Liger-Kernel monkey patches on the text backbone (requires liger-kernel).",
    )
    parser.add_argument(
        "--vlm-opt-torch-compile-model",
        action="store_true",
        help="Wrap the full model with torch.compile (experimental).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (lower activation VRAM; slower steps). Strong recommendation if OOM.",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="QLoRA: 4-bit NF4 base (bitsandbytes) + LoRA on LLM layers. Skips vlm_opt (fused). Vision sparsity defaults off; use --token-sparsity to enable.",
    )
    parser.add_argument(
        "--qlora-8bit",
        action="store_true",
        help="With --qlora, load base in 8-bit instead of 4-bit NF4.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default 16).")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default 32).")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default 0.05).",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Comma-separated module leaf names (e.g. q_proj,...,linear_fc1,linear_fc2 for projector). "
        "Default: LLM projections + vision merger/deepstack linear_fc1/linear_fc2.",
    )
    parser.add_argument(
        "--token-sparsity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Vision sparsity on get_image_features. Default: off with --qlora (full image tokens; better signal for LoRA), on without --qlora. "
        "Override with --token-sparsity / --no-token-sparsity.",
    )
    parser.add_argument(
        "--sparsity-keep-ratio",
        "--keep-ratio",
        type=float,
        default=0.5,
        metavar="R",
        dest="sparsity_keep_ratio",
        help="Vision sparsity: fraction of merged image tokens to keep per image (default 0.5). Values >= 1.0 keep all. Only applies when --token-sparsity.",
    )
    args = parser.parse_args()

    if args.token_sparsity is None:
        # QLoRA + vision sparsity often hurts convergence (fewer image tokens); default sparsity off for QLoRA.
        args.token_sparsity = not args.qlora

    if args.sparsity_keep_ratio <= 0:
        parser.error("--sparsity-keep-ratio / --keep-ratio must be positive.")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader-num-workers must be >= 0.")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1.")
    if args.vision_min_pixels > args.vision_max_pixels:
        parser.error("--vision-min-pixels must be <= --vision-max-pixels.")

    if args.qlora and not torch.cuda.is_available():
        parser.error("QLoRA requires CUDA (bitsandbytes GPU quantization).")
    if args.qlora_8bit and not args.qlora:
        parser.error("--qlora-8bit requires --qlora.")

    optim_cli = args.optim
    if optim_cli is None and args.qlora:
        optim_cli = "paged_adamw_8bit"

    effective_gc = args.gradient_checkpointing or args.qlora

    safe_print(
        "[finetune] CLI - "
        f"vlm_opt={args.vlm_opt}, qlora={args.qlora}, token_sparsity={args.token_sparsity}, "
        f"sparsity_keep_ratio={args.sparsity_keep_ratio}, "
        f"vlm_merger_backend={args.vlm_opt_merger_backend!r}, "
        f"vision_max_pixels={args.vision_max_pixels}, image_max_side={args.image_max_side}, "
        f"no_bf16={args.no_bf16}, gradient_checkpointing={effective_gc}, "
        f"dataloader_num_workers={args.dataloader_num_workers}"
    )

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.vision_min_pixels,
        max_pixels=args.vision_max_pixels,
    )
    if args.qlora:
        bnb = build_bitsandbytes_config(
            load_in_4bit=not args.qlora_8bit,
            bf16_compute=not args.no_bf16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            quantization_config=bnb,
            device_map="auto",
            cache_dir="./cache",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir="./cache",
        )

    use_token_sparsity = args.token_sparsity
    if use_token_sparsity:
        safe_print(
            "[finetune] Applying vision image sparsity on get_image_features "
            f"(keep_ratio={args.sparsity_keep_ratio}) ..."
        )
        model = apply_qwen3_sparsity(model, keep_ratio=args.sparsity_keep_ratio)

    use_vlm_opt = bool(args.vlm_opt) and not args.qlora
    if args.vlm_opt and args.qlora:
        safe_print(
            "[finetune] vlm_opt skipped: kernel fusion / compile conflicts with QLoRA quantized + PEFT wrap."
        )
    if use_vlm_opt:
        vlm_cfg = VlmOptConfig(
            enabled=True,
            fused_vision_merger=True,
            merger_fused_backend=MergerFusedBackend(args.vlm_opt_merger_backend),
            liger_language_model=args.vlm_opt_liger,
            torch_compile_full_model=args.vlm_opt_torch_compile_model,
        )
        model = apply_vlm_optimizations(model, vlm_cfg)

    if args.qlora:
        lora_targets = None
        if args.lora_target_modules:
            lora_targets = tuple(s.strip() for s in args.lora_target_modules.split(",") if s.strip())
        vram_cfg = VramTrainConfig(
            qlora=True,
            load_in_4bit=not args.qlora_8bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_targets,
        )
        model = apply_qlora_peft(model, vram_cfg, gradient_checkpointing=effective_gc)

    print("Loading dataset...")
    dataset = QwenDataset(
        dataset_id=args.dataset_id,
        split=args.split,
        processor=processor,
        cache_dir=args.cache_dir,
        image_max_side=args.image_max_side,
        shuffle=True,
        seed=args.seed,
    )

    run_finetune(
        model=model,
        processor=processor,
        dataset=dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bf16=False if args.no_bf16 else None,
        optim=optim_cli,
        seed=args.seed,
        gradient_checkpointing=effective_gc,
        qlora=args.qlora,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
