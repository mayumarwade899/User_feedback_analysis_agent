"""Agent that extracts structured technical details from bug-related feedback."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import ApplicationConfig
from ..schemas import BugInsights, GraphState
from ..utils import create_llm
from ..llm import parse_llm_json

logger = logging.getLogger(__name__)


class BugInsightsAgent:
    """Derives actionable bug information such as severity and reproduction context."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.llm = create_llm(config)

    def run(self, state: GraphState) -> GraphState:
        classification = state.get("classification")
        record = state.get("record")

        if not record or not classification or classification.category != "Bug":
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
                    "Bug insight extraction failed, using fallback: %s",
                    exc,
                )
                data = self._heuristic_fallback(record, classification)

        state["bug_insights"] = BugInsights(**data)
        return state

    def _heuristic_fallback(self, record, classification) -> dict:
        return {
            "severity": classification.priority,
            "steps_to_reproduce": "",
            "environment": record.payload.get("platform")
            or record.payload.get("subject", ""),
            "impact_summary": classification.rationale,
        }

    def _build_prompt(self, payload: dict) -> str:
        details = "\n".join(f"{k}: {v}" for k, v in payload.items())
        return (
            "Analyze the following user feedback and extract bug-related insights.\n"
            "Respond with a JSON object containing:\n"
            "severity, steps_to_reproduce, environment, impact_summary.\n\n"
            f"{details}"
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a senior QA engineer reviewing incoming bug reports.\n"
            "Summarize the issue in a way that helps engineers reproduce and fix it.\n"
            "If any detail is missing, use an empty string.\n"
            "Return only valid JSON."
        )
