"""Utility helpers for data loading, LLM access, and output handling."""
from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Mapping, Any

import pandas as pd
import requests
from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage

from .config import ApplicationConfig
from .schemas import FeedbackRecord

logger = logging.getLogger(__name__)


def load_input_records(config: ApplicationConfig) -> List[FeedbackRecord]:
    """Load feedback records from configured CSV sources."""
    records: List[FeedbackRecord] = []

    def sanitize(row: dict) -> dict:
        return {k: ("" if v is None else str(v)) for k, v in row.items()}

    sources = [
        (config.paths.app_reviews_file, "app_store", "review_id"),
        (config.paths.support_emails_file, "support_email", "email_id"),
    ]

    for file_path, source_type, id_field in sources:
        if not file_path.exists():
            logger.warning("Missing input file: %s", file_path)
            continue

        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            payload = sanitize(row.to_dict())
            source_id = payload.get(id_field)

            if not source_id:
                logger.debug("Skipping row without identifier in %s", file_path)
                continue

            records.append(
                FeedbackRecord(
                    source_id=source_id,
                    source_type=source_type,
                    payload=payload,
                )
            )

    return records


def prepare_output_dirs(config: ApplicationConfig) -> None:
    """Ensure output directories exist."""
    config.paths.output_root.mkdir(parents=True, exist_ok=True)


def write_csv_file(
    path: Path,
    rows: Iterable[dict],
    fieldnames: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class GeminiClient:
    """Lightweight Gemini wrapper exposing an invoke() method."""

    def __init__(self, model: str, temperature: float, max_tokens: int):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages: List[BaseMessage]) -> str:
        blocks: List[str] = []

        for msg in messages:
            role = getattr(msg, "type", "user")
            blocks.append(f"{role.upper()}:\n{msg.content}")

        prompt = "\n\n".join(blocks)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        return response.text


def create_llm(config: ApplicationConfig) -> Optional[GeminiClient]:
    """Instantiate Gemini LLM if enabled and configured."""
    if os.getenv("FEEDBACK_AUTOMATION_DISABLE_LLM") == "1":
        logger.info("LLM usage disabled via environment variable.")
        return None

    if config.llm.provider.lower() != "gemini":
        logger.info("Non-Gemini provider configured; skipping LLM.")
        return None

    try:
        return GeminiClient(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    except Exception as exc:
        logger.warning(
            "Gemini initialization failed (%s). Using heuristic mode.",
            exc,
        )
        return None


def generate_ticket_id(source_id: str, category: str) -> str:
    seed = f"{source_id}:{category}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"T-{digest.upper()}"


def stringify_messages(messages: List[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        role = getattr(msg, "type", getattr(msg, "role", "unknown"))
        lines.append(f"[{role}] {msg.content}")
    return "\n".join(lines)


def send_webhook(url: str, payload: Mapping[str, Any], timeout: int = 5) -> None:
    """Post ticket payload to external webhook (non-blocking)."""
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            logger.warning(
                "Webhook request failed: %s %s",
                resp.status_code,
                resp.text,
            )
    except Exception as exc:
        logger.exception("Webhook error: %s", exc)
