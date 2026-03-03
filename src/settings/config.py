"""Project configuration models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class Config(BaseModel):

    model_id: str = Field(default="Qwen/Qwen3-VL-8B-Instruct", description="Hugging Face model ID for the Qwen VL model.")
    dataset_id: str = Field(default="lmms-lab/ChartQA", description="Hugging Face dataset ID for the ChartQA dataset.")