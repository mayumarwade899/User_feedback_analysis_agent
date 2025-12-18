"""Agent responsible for extracting structured insights from feature requests."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import ApplicationConfig
from ..schemas import FeatureInsights, GraphState
from ..utils import create_llm
from ..llm import parse_llm_json

logger = logging.getLogger(__name__)


class FeatureInsightsAgent:
    """Converts feature-related feedback into actionable product insights."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.llm = create_llm(config)


    def run(self, state: GraphState) -> GraphState:
        classification = state.get("classification")
        record = state.get("record")

        if not record or not classification or classification.category != "Feature Request":
            return state

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=self._build_prompt(record.payload)),
        ]

        if self.llm is None:
            data = self._heuristic_fallback(record, classification)
        else:
            try:
                raw_response = self.llm.invoke(messages)
                data = parse_llm_json(raw_response)
            except Exception as exc:
                logger.warning(
                    "Feature insight extraction failed, using fallback: %s",
                    exc,
                )
                data = self._heuristic_fallback(record, classification)

        state["feature_insights"] = FeatureInsights(**data)
        return state


    def _heuristic_fallback(self, record, classification) -> dict:
        return {
            "user_value": "",
            "demand_level": classification.priority,
            "suggested_solution": (
                record.payload.get("review_text")
                or record.payload.get("body", "")
            ),
        }


    def _build_prompt(self, payload: dict) -> str:
        context = "\n".join(f"{k}: {v}" for k, v in payload.items())
        return (
            "Review the following user feedback and extract feature request insights.\n"
            "Respond with a JSON object containing:\n"
            "user_value, demand_level, suggested_solution.\n\n"
            f"{context}"
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a product manager reviewing incoming feature requests.\n"
            "Explain why the user wants the feature (user_value),\n"
            "estimate demand (Low, Medium, High, Critical),\n"
            "and suggest a concise implementation idea.\n"
            "Return only valid JSON."
        )
