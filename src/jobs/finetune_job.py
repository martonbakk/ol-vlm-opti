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

from src.data.data import QwenDataset


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
) -> None:
    """Run fine-tuning with model, processor and dataset already loaded (e.g. from notebook).

    ``bf16=None`` → enable bfloat16 when CUDA supports it (narrower activations = less memory traffic).
    ``optim=None`` → use fused AdamW on CUDA when available (fewer small GPU ops).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def collate(batch):
        return __collate_fn(batch, processor.tokenizer, pad_id)

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

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=5,
        bf16=use_bf16,
        fp16=False,
        optim=optim_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.add_callback(_ZeroGradSetToNoneCallback(trainer))

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


def __collate_fn(batch, tokenizer, pad_token_id: int = 0):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", default="test[:1%]")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--vision-max-pixels",
        type=int,
        default=262144,
        help="Upper bound for processor image pixels (H*W). Lower means less VRAM.",
    )
    parser.add_argument(
        "--vision-min-pixels",
        type=int,
        default=50176,
        help="Lower bound for processor image pixels (H*W).",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=768,
        help="Resize images on CPU so the longest side is at most this value.",
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
    args = parser.parse_args()

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.vision_min_pixels,
        max_pixels=args.vision_max_pixels,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir="./cache",
    )

    print("Loading dataset...")
    dataset = QwenDataset(
        dataset_id=args.dataset_id,
        split=args.split,
        processor=processor,
        cache_dir=args.cache_dir,
        image_max_side=args.image_max_side,
    )

    run_finetune(
        model=model,
        processor=processor,
        dataset=dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bf16=False if args.no_bf16 else None,
        optim=args.optim,
    )


if __name__ == "__main__":
    main()
