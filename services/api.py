from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from feedback_automation.config import AppConfig  # noqa: E402
from feedback_automation.graph import FeedbackPipeline  # noqa: E402
from feedback_automation.schemas import FeedbackRecord, ProcessingResult  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feedback Automation Service",
    version="1.1",
    description="API for classifying feedback and generating actionable tickets",
)


class FeedbackInput(BaseModel):
    message: str = Field(..., description="User feedback content")
    source_id: Optional[str] = Field(None, description="Unique feedback identifier")
    source_type: Optional[str] = Field("api", description="Origin of feedback")


class PipelineResponse(BaseModel):
    ticket: dict
    classification: dict
    bug_insights: Optional[dict] = None
    feature_insights: Optional[dict] = None
    quality: dict
    logs: list[dict]


@lru_cache(maxsize=1)
def get_pipeline() -> FeedbackPipeline:
    """
    Lazily initialize and cache the feedback pipeline.
    """
    logger.info("Initializing feedback pipeline for API usage")
    config = AppConfig.load()
    return FeedbackPipeline(config)


def _format_result(result: ProcessingResult) -> PipelineResponse:
    return PipelineResponse(
        ticket=result.ticket.model_dump(),
        classification=result.classification.model_dump(),
        bug_insights=result.bug_insights.model_dump() if result.bug_insights else None,
        feature_insights=result.feature_insights.model_dump() if result.feature_insights else None,
        quality=result.quality.model_dump(),
        logs=[log.model_dump() for log in result.logs],
    )


@app.post("/process", response_model=PipelineResponse)
def process_feedback(payload: FeedbackInput) -> PipelineResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Feedback message cannot be empty")

    pipeline = get_pipeline()

    record_payload = {
        "source_id": payload.source_id or "API_INPUT",
        "source_type": payload.source_type or "api",
        "feedback_text": payload.message,
        "review_text": payload.message,
        "body": payload.message,
    }

    try:
        result = pipeline.process_record(
            FeedbackRecord(
                source_id=record_payload["source_id"],
                source_type=record_payload["source_type"],
                payload=record_payload,
            )
        )
    except Exception as exc:
        logger.exception("API pipeline execution failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to process feedback",
        ) from exc

    return _format_result(result)
