"""Bounded CSV ingestion and spreadsheet-safe export shared by API clients."""

from __future__ import annotations

import csv
import io

import pandas as pd

from threadlens.config import Settings, settings
from threadlens.validation import has_visible_text


def parse_csv(content: bytes, config: Settings = settings) -> pd.DataFrame:
    """Validate before prediction. Preserve rows, string values, and input columns."""
    if len(content) > config.max_csv_bytes:
        raise ValueError(f"CSV exceeds the {config.max_csv_bytes // 1024 // 1024} MB limit.")
    if not content:
        raise ValueError("CSV is empty.")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must use UTF-8 encoding.") from exc
    if "\x00" in decoded:
        raise ValueError("CSV contains unsupported null bytes.")
    try:
        reader = csv.reader(io.StringIO(decoded), strict=True)
        columns = next(reader)
        if len(columns) != len(set(columns)):
            raise ValueError("CSV column names must be unique.")
        if "text" not in columns:
            raise ValueError("CSV must contain a column named 'text'.")
        records = []
        for row_number, row in enumerate(reader, start=1):
            if not row:
                raise ValueError(f"Row {row_number} is empty.")
            if len(row) != len(columns):
                raise ValueError(f"Row {row_number} has the wrong number of columns.")
            if row_number > config.max_csv_rows:
                raise ValueError(f"CSV exceeds the {config.max_csv_rows}-row limit.")
            text = row[columns.index("text")]
            if not has_visible_text(text):
                raise ValueError(f"Row {row_number} has empty text.")
            if len(text) > config.max_chars:
                raise ValueError(f"Row {row_number} exceeds {config.max_chars} characters.")
            records.append(row)
    except (csv.Error, StopIteration) as exc:
        raise ValueError("CSV is malformed or lacks a header.") from exc
    if not records:
        raise ValueError("CSV has no data rows.")
    return pd.DataFrame(records, columns=columns, dtype=str)


def validate_csv(content: bytes, config: Settings = settings) -> pd.DataFrame:
    return parse_csv(content, config)


def safe_cell(value: object) -> object:
    """Escape formulas even when a spreadsheet ignores leading whitespace."""
    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def safe_csv(frame: pd.DataFrame) -> bytes:
    escaped = frame.map(safe_cell)
    escaped.columns = [safe_cell(str(column)) for column in frame.columns]
    return escaped.to_csv(index=False).encode("utf-8-sig")
