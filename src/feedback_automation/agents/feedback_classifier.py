"""Agent responsible for classifying incoming user feedback."""
from __future__ import annotations

import logging
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import ApplicationConfig
from ..schemas import ClassificationOutcome, FeedbackRecord, GraphState
from ..utils import create_llm
from ..llm import parse_llm_json

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"Bug", "Feature Request", "Praise", "Complaint", "Spam"}
VALID_PRIORITIES = {"Critical", "High", "Medium", "Low"}
DEFAULT_PRIORITY_BY_CATEGORY = {
    "Bug": "High",
    "Feature Request": "Medium",
    "Praise": "Low",
    "Complaint": "Medium",
    "Spam": "Low",
}

class FeedbackClassifierAgent:
    """Determines feedback category, priority, and confidence."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.llm = create_llm(config)

    def _heuristic_classification(self, record: FeedbackRecord) -> Dict[str, str]:
        text = " ".join(record.payload.values()).lower()

        if any(k in text for k in ["crash", "error", "bug", "login", "data loss", "cannot", "failed"]):
            category = "Bug"
        elif any(k in text for k in ["please add", "would love", "feature", "dark mode", "add"]):
            category = "Feature Request"
        elif any(k in text for k in ["amazing", "love", "great", "awesome"]):
            category = "Praise"
        elif any(k in text for k in ["expensive", "slow", "poor"]):
            category = "Complaint"
        elif any(k in text for k in ["buy followers", "visit", "http", "spam"]):
            category = "Spam"
        else:
            category = "Complaint"

        return {
            "category": category,
            "priority": DEFAULT_PRIORITY_BY_CATEGORY.get(category, "Medium"),
            "confidence": 0.35,
            "rationale": "Heuristic keyword-based classification applied.",
        }

    def run(self, state: GraphState) -> GraphState:
        record = state.get("record")
        assert record, "FeedbackClassifierAgent requires a record"

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=self._build_prompt(record)),
        ]

        if self.llm is None:
            parsed = self._heuristic_classification(record)
        else:
            try:
                raw_response = self.llm.invoke(messages)
                parsed = parse_llm_json(raw_response)
            except Exception as exc:
                logger.warning(
                    "LLM classification failed, falling back to heuristics: %s",
                    exc,
                )
                parsed = self._heuristic_classification(record)

        category = parsed.get("category", "Complaint")
        if category not in VALID_CATEGORIES:
            logger.debug("Unknown category '%s', defaulting to Complaint", category)
            category = "Complaint"

        priority = parsed.get("priority") or DEFAULT_PRIORITY_BY_CATEGORY.get(category, "Medium")
        raw_confidence = parsed.get("confidence", 0.5)

        if isinstance(raw_confidence, (int, float)):
            confidence = float(raw_confidence)
        elif isinstance(raw_confidence, str):
            confidence_map = {
                "high": 0.85,
                "medium": 0.6,
                "low": 0.35,
            }
            confidence = confidence_map.get(raw_confidence.lower(), 0.5)
        else:
            confidence = 0.5
        rationale = parsed.get("rationale", "Classification completed")

        state["classification"] = ClassificationOutcome(
            category=category,
            priority=priority,
            confidence=confidence,
            rationale=rationale,
        )

        return state

    def _build_prompt(self, record: FeedbackRecord) -> str:
        content = "\n".join(f"{k}: {v}" for k, v in record.payload.items())
        return (
            "Analyze the following user feedback and classify it.\n"
            "Return a JSON object with keys: category, priority, confidence, rationale.\n\n"
            f"{content}"
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a product feedback triage assistant.\n"
            "Classify feedback into: Bug, Feature Request, Praise, Complaint, or Spam.\n"
            "Assign an appropriate priority and provide a short rationale.\n"
            "Respond strictly in valid JSON."
        )
