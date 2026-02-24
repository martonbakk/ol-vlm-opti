from datasets import load_dataset

from ..config import AppConfig
from ..data import QwenDataset
from ..models import load_model, load_processor
from ..training import trainer, training_arguments


def run_training(config: AppConfig | None = None):
    app_config = config or AppConfig()

    print("Nsight Profilozás Indul: Qwen 3 VL QLoRA Tanítás...")

    processor = load_processor(app_config.model)
    model = load_model(app_config.model, app_config.lora)

    chart_dataset = load_dataset(
        app_config.dataset.dataset_id,
        split=app_config.dataset.split,
    )
    train_dataset = QwenDataset(chart_dataset, processor)

    args = training_arguments(app_config.trainer)
    train_loop = trainer(
        model=model,
        args=args,
        dataset=train_dataset,
    )

    train_loop.train()
    print("✅ Ciklus vége, Nsight adatok rögzítve!")
