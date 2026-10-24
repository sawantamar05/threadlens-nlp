"""Central paths and bounded service settings. No global environment changes."""

import os
from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("analysis", "request", "commentary")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
DEFINITIONS = {
    "analysis": "A substantive claim supported by a specific observation, comparison, measurement, quotation, or explanatory mechanism.",
    "request": "A genuine primary request for information, clarification, advice, validation, or action.",
    "commentary": "An opinion, reaction, joke, unsupported assertion, or resource share without a primary request or supported claim.",
}
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = field(default_factory=lambda: Path(os.getenv("THREADLENS_ROOT", str(ROOT))))
    max_chars: int = 10000
    max_batch: int = 100
    max_csv_bytes: int = 2 * 1024 * 1024
    max_csv_rows: int = 1000
    low_confidence: float = 0.6

    @property
    def local(self) -> Path:
        return self.root / ".local"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def database(self) -> Path:
        return self.local / "threadlens.sqlite3"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


settings = Settings()
