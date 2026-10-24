"""Public, validated API schema. Labels have one canonical ordering."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from threadlens.validation import has_visible_text

Label = Literal["analysis", "request", "commentary"]
CommentText = Annotated[str, Field(min_length=1, max_length=10000)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionRequest(InputModel):
    text: CommentText
    model_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not has_visible_text(value):
            raise ValueError("Comment text must not be empty or whitespace.")
        if "\x00" in value:
            raise ValueError("Comment text must not contain null bytes.")
        return value


class BatchRequest(InputModel):
    texts: list[CommentText] = Field(min_length=1, max_length=100)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("texts")
    @classmethod
    def nonempty(cls, values: list[str]) -> list[str]:
        if any(not has_visible_text(value) for value in values):
            raise ValueError("Each comment must contain non-whitespace text.")
        if any("\x00" in value for value in values):
            raise ValueError("Comment text must not contain null bytes.")
        return values


class Prediction(BaseModel):
    label: Label
    scores: dict[Label, float]
    model_id: str
    model_version: str
    truncated: bool
    processing_ms: float
    warnings: list[str]
    low_confidence: bool


class BatchResponse(BaseModel):
    results: list[Prediction]
    processing_ms: float


class ThreadRequest(InputModel):
    story: str | int
    limit: int = Field(default=30, ge=1, le=100)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("story", mode="before")
    @classmethod
    def story_identifier(cls, value: str | int) -> str | int:
        from threadlens.data.hn import parse_story_id

        if type(value) not in {str, int}:
            raise ValueError("Story must be an integer ID or supported Hacker News URL.")
        parse_story_id(value)
        return value


class ThreadComment(Prediction):
    comment_id: int
    thread_id: int
    text: str
    permalink: str
    reviewed_label: Label | None = None
    reviewed_provenance: str | None = None


class ThreadResponse(BaseModel):
    story_id: int
    title: str
    comments: list[ThreadComment]
    counts: dict[Label, int]
    proportions: dict[Label, float]
    warnings: list[str]


class ModelInfo(BaseModel):
    id: str
    name: str
    kind: str
    version: str
    provenance: str
    available: bool
    metadata: dict[str, Any]


class ModelsResponse(BaseModel):
    active_model: str | None
    models: list[ModelInfo]


class AnnotationRequest(InputModel):
    comment_id: str = Field(min_length=1, max_length=100)
    label: Label
    provenance: Literal["human_independent", "ai_assisted_reviewed"]
    blind: bool = False
    notes: str = Field(default="", max_length=2000)


class AnnotationResponse(BaseModel):
    revision: int
    comment_id: str
    label: Label
    provenance: str
    review_status: str
    timestamp: str
    notes: str
    blind: int
    independent_initial: int
