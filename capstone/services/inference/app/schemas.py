from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CHARS = 1000


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text tidak boleh kosong/spasi saja")
        return v


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Message] = Field(min_length=1)


class Classification(BaseModel):
    spam_probability: float
    is_spam: bool


class ClassifyResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Classification]
