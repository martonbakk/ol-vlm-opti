import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ..config import LoraTrainingConfig, ModelConfig


def load_processor(config: ModelConfig):
    return AutoProcessor.from_pretrained(config.model_id)


def load_model(model_config: ModelConfig, lora_config: LoraTrainingConfig):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=list(lora_config.target_modules),
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )
    model = get_peft_model(model, peft_config)
    return model
