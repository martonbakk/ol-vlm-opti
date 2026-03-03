"""Dataset for vision-language model fine-tuning."""

from typing import Any

import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class QwenDataset(Dataset):
    """ChartQA dataset wrapped for Qwen VL chat format."""

    def __init__(
        self,
        dataset_id: str,
        split: str,
        processor: Any,
        cache_dir: str | None = None,
    ) -> None:
        self.dataset = load_dataset(dataset_id, split=split, cache_dir=cache_dir)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        return self.__to_model_inputs(item)

    def __to_model_inputs(self, item: dict[str, Any]) -> dict[str, Any]:
        image_data = item["image"]
        image = (
            Image.fromarray(np.array(image_data, dtype=np.uint8))
            if isinstance(image_data, (list, np.ndarray))
            else image_data
        )
        question = item["question"]
        answer = str(item["answer"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
