"""Agent responsible for constructing issue tickets from processed feedback."""
from __future__ import annotations

import textwrap

from ..config import ApplicationConfig
from ..schemas import GraphState, TicketData
from ..utils import generate_ticket_id


class TicketBuilderAgent:
    """Assembles a structured ticket using classification and insight outputs."""

    def __init__(self, config: ApplicationConfig):
        self.config = config


    def run(self, state: GraphState) -> GraphState:
        record = state.get("record")
        classification = state.get("classification")

        if not record or not classification:
            return state

        bug_insights = state.get("bug_insights")
        feature_insights = state.get("feature_insights")

        metadata = self._build_metadata(record, bug_insights, feature_insights)

        ticket = TicketData(
            ticket_id=self._generate_ticket_id(record, classification),
            title=self._generate_title(record, classification),
            description=self._generate_description(
                record,
                classification,
                bug_insights,
                feature_insights,
            ),
            category=classification.category,
            priority=classification.priority,
            source_id=record.source_id,
            source_type=record.source_type,
            metadata=metadata,
        )

        state["ticket"] = ticket
        return state


    def _generate_ticket_id(self, record, classification) -> str:
        return generate_ticket_id(record.source_id, classification.category)

    def _build_metadata(self, record, bug_insights, feature_insights) -> dict:
        meta = {
            "source_id": record.source_id,
            "source_type": record.source_type,
            "payload_fields": ",".join(record.payload.keys()),
        }

        if bug_insights:
            meta.update(
                {
                    "bug_severity": bug_insights.severity or "",
                    "bug_environment": bug_insights.environment or "",
                }
            )

        if feature_insights:
            meta["feature_demand"] = feature_insights.demand_level or ""

        return meta

    def _generate_title(self, record, classification) -> str:
        if record.source_type == "app_store":
            base_text = record.payload.get("review_text", "")
        else:
            base_text = record.payload.get("subject", "")

        base_text = base_text.strip()

        if base_text:
            return base_text.split(".")[0][:100]

        return f"{classification.category} feedback from {record.source_type}"

    def _generate_description(
        self,
        record,
        classification,
        bug_insights,
        feature_insights,
    ) -> str:
        sections = [
            f"Category: {classification.category}",
            f"Priority: {classification.priority}",
            f"Confidence Score: {classification.confidence:.2f}",
            "",
            "Original Feedback:",
            record.payload.get("review_text")
            or record.payload.get("body")
            or "",
        ]

        if bug_insights:
            sections.extend(
                [
                    "",
                    "Bug Details:",
                    f"- Severity: {bug_insights.severity or 'Unspecified'}",
                    f"- Environment: {bug_insights.environment or 'Unknown'}",
                    f"- Steps to Reproduce: {bug_insights.steps_to_reproduce or 'Not provided'}",
                    f"- Impact Summary: {bug_insights.impact_summary or 'Not stated'}",
                ]
            )

        if feature_insights:
            sections.extend(
                [
                    "",
                    "Feature Insights:",
                    f"- User Value: {feature_insights.user_value or 'Unspecified'}",
                    f"- Demand Level: {feature_insights.demand_level or 'Unknown'}",
                    f"- Suggested Solution: {feature_insights.suggested_solution or 'Not provided'}",
                ]
            )

        sections.extend(
            [
                "",
                "Source Information:",
                f"- Source Type: {record.source_type}",
                f"- Payload Fields: {', '.join(record.payload.keys())}",
            ]
        )

        return textwrap.dedent("\n".join(sections)).strip()
