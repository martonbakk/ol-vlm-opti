from pathlib import Path
import os

from dotenv import dotenv_values, load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

load_dotenv()


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    vl_model_id: str = Field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        description="Hugging Face model ID for the Qwen VL model (smaller default for dev/finetune).",
    )

    jg_model_id: str = Field(
        default="qwen/qwen3-32b",
        description="Groq model ID used by the deepeval judge model.",
    )

    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key loaded from GROQ_API_KEY.",
    )

    groq_reasoning_effort: str | None = Field(
        default="high",
        description="Optional Groq reasoning effort parameter.",
    )

    groq_reasoning_format: str | None = Field(
        default=None,
        description="Optional Groq reasoning format parameter.",
    )

    judge_temperature: float = Field(
        default=0.1,
        description="Temperature for judge model generation.",
    )

    dataset_id: str = Field(
        default="lmms-lab/ChartQA",
        description="Hugging Face dataset ID for the ChartQA dataset.",
    )

    deepeval_api_key: str | None = Field(
        default=None,
        description="API key for DeepEval, loaded from DEEPEVAL_API_KEY.",
    )