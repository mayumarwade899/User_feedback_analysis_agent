"""Agent that evaluates generated tickets for readiness and clarity."""
from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import ApplicationConfig
from ..schemas import GraphState, QualityAssessment
from ..utils import create_llm
from ..llm import parse_llm_json

logger = logging.getLogger(__name__)


class TicketQualityAgent:
    """Performs final validation checks on generated issue tickets."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.llm = create_llm(config)


    def run(self, state: GraphState) -> GraphState:
        ticket = state.get("ticket")
        if not ticket:
            return state

        messages = [
            SystemMessage(content=self._system_instruction()),
            HumanMessage(content=self._build_review_prompt(ticket.description)),
        ]

        if self.llm is None:
            review_payload = self._rule_based_review(ticket)
        else:
            try:
                response = self.llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                review_payload = parse_llm_json(content)
            except Exception as exc:
                logger.warning(
                    "Ticket quality evaluation failed, using fallback: %s",
                    exc,
                    exc_info=False,
                )
                review_payload = self._rule_based_review(ticket)

        state["quality"] = QualityAssessment(**review_payload)
        return state

    def _rule_based_review(self, ticket) -> dict:
        problems: List[str] = []

        if not ticket.title:
            problems.append("Ticket title is missing")

        if not ticket.description:
            problems.append("Ticket description is missing")

        return {
            "passed": len(problems) == 0,
            "issues": problems,
            "suggestions": (
                "Provide a concise title and ensure the description contains enough context."
                if problems
                else "No improvements required."
            ),
        }


    def _build_review_prompt(self, description: str) -> str:
        return (
            "Evaluate the following issue description and decide whether it is ready"
            " to be handed off to an engineering team.\n\n"
            "Return a strict JSON object with the following fields:\n"
            "- passed (boolean)\n"
            "- issues (array of strings)\n"
            "- suggestions (string)\n\n"
            f"Issue Description:\n{description}"
        )

    @staticmethod
    def _system_instruction() -> str:
        return (
            "You are an experienced QA lead reviewing issue tickets."
            " Focus on clarity, completeness, and actionability."
            " If critical information is missing, passed must be false."
            " Respond strictly in JSON format."
        )
