import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class QwenDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
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
