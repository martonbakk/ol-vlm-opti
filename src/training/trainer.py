import torch
from transformers import Trainer, TrainingArguments

from ..config import TrainerConfig


def collate_batch(features):
    batch = {}
    for key in features[0].keys():
        if torch.is_tensor(features[0][key]):
            batch[key] = torch.stack([feature[key] for feature in features])
    return batch


def training_arguments(config: TrainerConfig):
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        max_steps=config.max_steps,
        remove_unused_columns=config.remove_unused_columns,
        report_to=config.report_to,
    )


def trainer(model, args, dataset):
    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_batch,
    )
