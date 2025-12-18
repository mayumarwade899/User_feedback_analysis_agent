"""LangGraph orchestration for the feedback automation system."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable, List

import pandas as pd
from langgraph.graph import END, StateGraph

from .agents import (
    BugInsightsAgent,
    FeatureInsightsAgent,
    FeedbackClassifierAgent,
    TicketQualityAgent,
    TicketBuilderAgent,
)
from .config import ApplicationConfig
from .schemas import (
    FeedbackRecord,
    GraphState,
    PipelineLog,
    PipelineMetrics,
    PipelineResult,
)
from .utils import (
    prepare_output_dirs,
    load_input_records,
    write_csv_file,
)

logger = logging.getLogger(__name__)


class FeedbackPipeline:
    """Coordinates the end-to-end feedback processing workflow."""

    def __init__(self, config: ApplicationConfig | None = None) -> None:
        self.config = config or ApplicationConfig.load()

        self.feedback_classifier = FeedbackClassifierAgent(self.config)
        self.bug_insights_agent = BugInsightsAgent(self.config)
        self.feature_insights_agent = FeatureInsightsAgent(self.config)
        self.ticket_builder_agent = TicketBuilderAgent(self.config)
        self.ticket_quality_agent = TicketQualityAgent(self.config)

        self._state_graph = self._build_graph()
        self.graph = self._state_graph.compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GraphState)

        graph.set_entry_point("classify")

        graph.add_node("classify", self._with_logging(self._classify))
        graph.add_node("bug_analysis", self._with_logging(self._analyze_bug))
        graph.add_node("feature_extract", self._with_logging(self._extract_feature))
        graph.add_node("ticket", self._with_logging(self._create_ticket))
        graph.add_node("quality", self._with_logging(self._quality_check))

        graph.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "bug": "bug_analysis",
                "feature": "feature_extract",
                "other": "ticket",
            },
        )

        graph.add_edge("bug_analysis", "ticket")
        graph.add_edge("feature_extract", "ticket")
        graph.add_edge("ticket", "quality")
        graph.add_edge("quality", END)

        return graph

    def _with_logging(
        self,
        fn: Callable[[GraphState], GraphState],
    ) -> Callable[[GraphState], GraphState]:

        def wrapped(state: GraphState) -> GraphState:
            logs = state.get("logs", []) or []
            record = state.get("record")
            source_id = record.source_id if record else "unknown"

            logger.info("Starting step %s for %s", fn.__name__, source_id)
            before = len(logs)

            updated = fn(state)
            after = len(updated.get("logs", []))

            if after == before:
                logs.append(
                    PipelineLog(
                        source_id=source_id,
                        step=fn.__name__,
                        status="completed",
                        details="Step executed successfully",
                    )
                )
                updated["logs"] = logs

            logger.info("Completed step %s for %s", fn.__name__, source_id)
            return updated

        return wrapped

    def _classify(self, state: GraphState) -> GraphState:
        updated = self.feedback_classifier.run(state)
        classification = updated.get("classification")

        logs = updated.get("logs", []) or []
        logs.append(
            PipelineLog(
                source_id=state["record"].source_id,
                step="classification",
                status="completed",
                details=f"Category={classification.category}, Priority={classification.priority}",
            )
        )
        updated["logs"] = logs
        return updated

    def _analyze_bug(self, state: GraphState) -> GraphState:
        updated = self.bug_insights_agent.run(state)
        logs = updated.get("logs", []) or []
        logs.append(
            PipelineLog(
                source_id=state["record"].source_id,
                step="bug_analysis",
                status="completed",
                details="Bug insights extracted",
            )
        )
        updated["logs"] = logs
        return updated

    def _extract_feature(self, state: GraphState) -> GraphState:
        updated = self.feature_insights_agent.run(state)
        logs = updated.get("logs", []) or []
        logs.append(
            PipelineLog(
                source_id=state["record"].source_id,
                step="feature_extraction",
                status="completed",
                details="Feature insights extracted",
            )
        )
        updated["logs"] = logs
        return updated

    def _create_ticket(self, state: GraphState) -> GraphState:
        updated = self.ticket_builder_agent.run(state)
        logs = updated.get("logs", []) or []
        logs.append(
            PipelineLog(
                source_id=state["record"].source_id,
                step="ticket_creation",
                status="completed",
                details=f"Ticket={updated['ticket'].ticket_id}",
            )
        )
        updated["logs"] = logs
        return updated

    def _quality_check(self, state: GraphState) -> GraphState:
        updated = self.ticket_quality_agent.run(state)
        quality = updated.get("quality")

        logs = updated.get("logs", []) or []
        logs.append(
            PipelineLog(
                source_id=state["record"].source_id,
                step="quality_review",
                status="passed" if quality and quality.passed else "issues",
                details=", ".join(quality.issues) if quality and quality.issues else "OK",
            )
        )
        updated["logs"] = logs
        return updated

    def _route_after_classification(self, state: GraphState) -> str:
        classification = state.get("classification")

        if not classification:
            return "other"

        if classification.category == "Bug":
            return "bug"

        if classification.category == "Feature Request":
            return "feature"

        return "other"
    
    def mermaid_diagram(self) -> str:
        """Return Mermaid diagram for visualization."""
        return self.graph.get_graph().draw_mermaid()
    
    def ascii_diagram(self) -> str:
        """Return ASCII depiction of the graph."""
        return self.graph.get_graph().draw_ascii()


    def process(self) -> List[PipelineResult]:
        prepare_output_dirs(self.config)

        results: List[PipelineResult] = []
        logger.info("Starting feedback processing run.")

        for record in load_input_records(self.config):
            results.append(self.process_record(record))

        logger.info("Processing complete. Generated %d tickets.", len(results))
        return results

    def process_record(self, record: FeedbackRecord) -> PipelineResult:
        state: GraphState = {"record": record, "logs": []}
        final_state = self.graph.invoke(state)
        return self._state_to_result(final_state)

    @staticmethod
    def _state_to_result(state: GraphState) -> PipelineResult:
        return PipelineResult(
            ticket=state["ticket"],
            quality=state["quality"],
            logs=state.get("logs", []),
            classification=state.get("classification"),
            bug_insights=state.get("bug_insights"),
            feature_insights=state.get("feature_insights"),
        )

    def write_outputs(self, results: List[PipelineResult]) -> None:
        tickets_rows = []
        log_rows = []

        for item in results:
            tickets_rows.append(
                {
                    "ticket_id": item.ticket.ticket_id,
                    "title": item.ticket.title,
                    "description": item.ticket.description,
                    "category": item.ticket.category,
                    "priority": item.ticket.priority,
                    "source_id": item.ticket.source_id,
                    "source_type": item.ticket.source_type,
                    "metadata": json.dumps(item.ticket.metadata),
                }
            )

            for log in item.logs:
                log_rows.append(
                    {
                        "source_id": log.source_id,
                        "step": log.step,
                        "status": log.status,
                        "details": log.details,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        write_csv_file(
            self.config.paths.tickets_output_file,
            tickets_rows,
            [
                "ticket_id",
                "title",
                "description",
                "category",
                "priority",
                "source_id",
                "source_type",
                "metadata",
            ],
        )

        write_csv_file(
            self.config.paths.pipeline_log_file,
            log_rows,
            ["source_id", "step", "status", "details", "timestamp"],
        )

        metrics = self._calculate_metrics(results)
        write_csv_file(
            self.config.paths.metrics_file,
            [metrics],
            ["processed", "by_category", "by_priority", "avg_confidence"],
        )

    def _calculate_metrics(self, results: List[PipelineResult]) -> dict:
        processed = len(results)
        by_category: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        total_conf = 0.0

        for result in results:
            by_category[result.classification.category] = (
                by_category.get(result.classification.category, 0) + 1
            )
            by_priority[result.classification.priority] = (
                by_priority.get(result.classification.priority, 0) + 1
            )
            total_conf += result.classification.confidence

        avg_confidence = total_conf / processed if processed else 0.0

        return {
            "processed": processed,
            "by_category": by_category,
            "by_priority": by_priority,
            "avg_confidence": round(avg_confidence, 3),
        }
