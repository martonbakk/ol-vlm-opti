import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class QwenWrapper:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir="./cache",
        )

    def __build_messages(self, question: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def __prepare_chat_inputs(self, image, question: str):
        messages = self.__build_messages(question)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.processor(
            text=[prompt],
            images=[image.convert("RGB")],
            return_tensors="pt",
        ).to(self.model.device)

    def answer(self, image, question: str, max_new_tokens: int = 128) -> str:
        inputs = self.__prepare_chat_inputs(image=image, question=question)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[-1] :]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def generate(self, image, question: str, max_length: int = 512) -> str:
        return self.answer(image=image, question=question, max_new_tokens=max_length)