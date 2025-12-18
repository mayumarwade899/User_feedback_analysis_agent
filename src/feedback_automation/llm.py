"""Utilities for handling and validating LLM responses."""
from __future__ import annotations

import logging
from typing import Any, Dict

import json5
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


def _remove_code_wrappers(text: str) -> str:
    """
    Remove markdown-style code fences and language hints from LLM output.
    """
    if not text:
        return ""

    value = text.strip()

    if value.startswith("```"):
        value = value.removeprefix("```").strip()

        if value.lower().startswith("json"):
            value = value[4:].strip()

        if value.endswith("```"):
            value = value.removesuffix("```").strip()

    return value.strip("` \n")


def parse_llm_json(raw_output: str) -> Dict[str, Any]:
    """
    Parse JSON returned by an LLM.
    Attempts direct parsing first, then retries after cleaning formatting noise.
    """
    if not raw_output:
        raise ValueError("LLM response is empty")

    try:
        return json5.loads(raw_output)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Direct JSON parse failed, attempting cleanup: %s", exc)

    cleaned_output = _remove_code_wrappers(raw_output)

    if not cleaned_output:
        raise ValueError("LLM response could not be cleaned into valid JSON")

    try:
        return json5.loads(cleaned_output)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse cleaned LLM output: %s", exc)
        raise


def summarize_messages(messages: list[BaseMessage]) -> str:
    """
    Convert a list of LangChain messages into a readable debug string.
    """
    return "\n".join(
        f"[{msg.type.upper()}] {getattr(msg, 'content', '')}"
        for msg in messages
    )
