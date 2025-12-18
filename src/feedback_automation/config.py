"""Application configuration management for the feedback automation project."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ProjectPaths:
    """Defines all input and output file locations."""

    data_root: Path = Path("data")
    output_root: Path = Path("outputs")

    app_reviews_file: Path = Path("data/reviews.csv")
    support_emails_file: Path = Path("data/emails.csv")
    expected_labels_file: Path = Path("data/expected_classifications.csv")

    tickets_output_file: Path = Path("outputs/generated_tickets.csv")
    pipeline_log_file: Path = Path("outputs/processing_log.csv")
    metrics_file: Path = Path("outputs/metrics.csv")


@dataclass
class ModelConfig:
    """Configuration for the language model provider."""

    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 1500


@dataclass
class ClassificationRules:
    """Thresholds and heuristics used during classification."""

    confidence_threshold: float = 0.35

    priority_map: Dict[str, str] = field(
        default_factory=lambda: {
            "Bug": "High",
            "Feature Request": "Medium",
            "Praise": "Low",
            "Complaint": "Medium",
            "Spam": "Low",
        }
    )

    critical_keywords: Dict[str, str] = field(
        default_factory=lambda: {
            "data loss": "Critical",
            "crash": "Critical",
            "cannot login": "High",
            "oauth": "High",
        }
    )


@dataclass
class ApplicationConfig:
    """Top-level configuration container."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    llm: ModelConfig = field(default_factory=ModelConfig)
    classification: ClassificationRules = field(default_factory=ClassificationRules)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ApplicationConfig":
        default_path = Path("config/config.yaml")

        if path is None and default_path.exists():
            path = default_path

        if path is None or not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ApplicationConfig":
        def hydrate(dc_type, values: Dict[str, Any]):
            allowed = {f.name for f in dc_type.__dataclass_fields__.values()}
            cleaned = {k: v for k, v in values.items() if k in allowed}

            for key, val in cleaned.items():
                if key.endswith("_file") or key.endswith("_root"):
                    cleaned[key] = Path(val)

            return dc_type(**cleaned)

        return cls(
            paths=hydrate(ProjectPaths, payload.get("paths", {})),
            llm=hydrate(ModelConfig, payload.get("llm", {})),
            classification=hydrate(
                ClassificationRules, payload.get("classification", {})
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": {
                "data_root": str(self.paths.data_root),
                "output_root": str(self.paths.output_root),
                "app_reviews_file": str(self.paths.app_reviews_file),
                "support_emails_file": str(self.paths.support_emails_file),
                "expected_labels_file": str(self.paths.expected_labels_file),
                "tickets_output_file": str(self.paths.tickets_output_file),
                "pipeline_log_file": str(self.paths.pipeline_log_file),
                "metrics_file": str(self.paths.metrics_file),
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "classification": {
                "confidence_threshold": self.classification.confidence_threshold,
                "priority_map": self.classification.priority_map,
                "critical_keywords": self.classification.critical_keywords,
            },
            "webhook": "https://jira.com/",
        }

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                json.loads(json.dumps(self.to_dict())),
                f,
                sort_keys=False,
            )
