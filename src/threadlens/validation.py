"""Input rules shared by the API, CLI inference service, and CSV ingestion."""

from __future__ import annotations

from unicodedata import category

from threadlens.config import Settings


class InputInvalid(ValueError):
    """A bounded application input failed validation."""


def has_visible_text(text: str) -> bool:
    return any(not char.isspace() and category(char) not in {"Cf", "Cc", "Cs"} for char in text)


def validate_texts(texts: list[str], config: Settings) -> None:
    if not isinstance(texts, list) or not 1 <= len(texts) <= config.max_batch:
        raise InputInvalid(f"Provide between 1 and {config.max_batch} comments.")
    for index, text in enumerate(texts, start=1):
        if not isinstance(text, str) or not has_visible_text(text):
            raise InputInvalid(f"Comment {index} must contain visible text.")
        if "\x00" in text:
            raise InputInvalid(f"Comment {index} must not contain null bytes.")
        if len(text) > config.max_chars:
            raise InputInvalid(f"Comment {index} exceeds {config.max_chars} characters.")
