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
        image_max_side: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = load_dataset(dataset_id, split=split, cache_dir=cache_dir)
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.processor = processor
        self.image_max_side = image_max_side

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
        if self.image_max_side is not None:
            image = self.__resize_image_keep_aspect(image, self.image_max_side)
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
        # Labels are built once per batch in finetune_job.__collate_fn (fewer clones / D2D than per-sample).
        return inputs

    @staticmethod
    def __resize_image_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
        width, height = image.size
        longest_side = max(width, height)
        if longest_side <= max_side:
            return image

        scale = max_side / float(longest_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height), Image.BICUBIC)
