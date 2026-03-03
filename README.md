# ol-vlm-opti

Vision-Language Model fine-tuning with Qwen3-VL-8B + QLoRA (ChartQA).

## Quick Start

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
