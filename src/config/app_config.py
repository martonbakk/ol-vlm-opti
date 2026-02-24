from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str = "lmms-lab/ChartQA"
    split: str = "test[:1%]"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass(frozen=True)
class LoraTrainingConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str = "./qwen_chartqa_lora"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 1
    learning_rate: float = 2e-4
    bf16: bool = True
    max_steps: int = 10
    remove_unused_columns: bool = False
    report_to: str = "none"


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraTrainingConfig = field(default_factory=LoraTrainingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
