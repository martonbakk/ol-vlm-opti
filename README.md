# ol-vlm-opti

Vision-Language Model fine-tuning with Qwen3-VL-8B + QLoRA (ChartQA).

## Quick Start


$ export PATH="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.1/target-windows-x64:$PATH"


```bash
# Install (uv)
uv sync

# Train
uv run python train.py
```

Checkpoints are saved to `./checkpoints/` by default.

## Project Structure

```
ol-vlm-opti/
├── train.py           # Entry point
├── src/
│   ├── config.py      # Config dataclasses (dataset, model, LoRA, trainer)
│   ├── data.py        # QwenDataset for ChartQA
│   ├── model.py       # Model + processor loading (QLoRA)
│   └── train.py       # Training pipeline
├── scripts/
│   ├── run_with_profile.sh   # Nsight profiling + GPU monitor
│   └── monitor.sh            # Standalone GPU monitor
└── pyproject.toml
```

## Configuration

Edit `src/config.py` to change:

- **Dataset**: `lmms-lab/ChartQA`, split `test[:1%]`
- **Model**: `Qwen/Qwen3-VL-8B-Instruct`, 4-bit NF4
- **LoRA**: r=16, alpha=32, modules `q_proj`, `v_proj`
- **Training**: batch size, steps, learning rate, output dir

## Profiling (Linux)

```bash
./scripts/run_with_profile.sh        # runs train.py by default
./scripts/run_with_profile.sh train.py
```

Requires Nsight Systems and `nvidia-smi`.

### Nsight: D2D / memset (finetune job)

If the report shows lots of **device-to-device copy** or **memset**:

- **`finetune_job`** builds **labels once per batch** (not per sample) and keeps tensors **contiguous** before the forward pass where possible.
- Training uses **`zero_grad(set_to_none=True)`** on the model (via callback) so gradients are dropped without zero-filling tensors when possible.
- **BF16** is enabled automatically when the GPU supports it (less traffic than FP32). Use CLI **`--no-bf16`** to turn off.
- On CUDA, **fused AdamW** (`adamw_torch_fused`) is the default; if it errors on your stack, pass **`--optim adamw_torch`**.

## VLM backend optimizations (`src/vlm_opt`)

Switchable stack for **kernel fusion** and **runtime module swap** (course: projector + optional Liger text kernels).

```python
from src.vlm_opt import VlmOptConfig, apply_vlm_optimizations

cfg = VlmOptConfig(
    enabled=True,
    fused_vision_merger=True,
    merger_fused_backend="auto",  # triton | pytorch | compile_wrap | auto
    liger_language_model=False,    # optional: needs `liger-kernel`
    torch_compile_full_model=False,
)
model = ...  # loaded Qwen3-VL (e.g. AutoModelForImageTextToText)
model = apply_vlm_optimizations(model, cfg)
```

- **`merger_fused_backend`**: replaces vision `PatchMerger` **linear_fc1 + GELU** with `FusedLinearGELU` (same weights on `linear_fc1`). **PyTorch** path is always safe for training. **Triton** custom kernel runs **eval-only** (no autograd through Triton); training uses eager PyTorch. On **Windows**, prefer **`compile_wrap`** or **`pytorch`** if Triton is flaky — set `VLM_OPT_DISABLE_TRITON=1` to force PyTorch.
- **`liger_language_model`**: tries `liger_kernel` monkey patches when the package is installed (API varies by version).
- **`torch_compile_full_model`**: wraps the full model in `torch.compile` (experimental; use `mode="reduce-overhead"` for CUDA graphs on Windows where supported).

Optional installs: `pip install triton` (platform-dependent), `pip install liger-kernel`.

**VRAM:** `--vlm-opt` does **not** guarantee lower peak VRAM (merger fusion saves traffic/overhead, not model weights). Prefer **`--gradient-checkpointing`**, smaller **`--batch-size`**, lower **`--vision-max-pixels`** / **`--image-max-side`**, keep **`bf16`** (omit `--no-bf16`), and/or a smaller **`--model-id`**. Logs prefixed with **`[vlm_opt]`** and **`[finetune]`** show what actually activated.
