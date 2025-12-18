"""Core data models shared across the feedback automation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class FeedbackCategory(str):
    BUG = "Bug"
    FEATURE_REQUEST = "Feature Request"
    PRAISE = "Praise"
    COMPLAINT = "Complaint"
    SPAM = "Spam"


class FeedbackRecord(BaseModel):
    source_id: str
    source_type: str
    payload: Dict[str, str]


class ClassificationOutcome(BaseModel):
    category: str
    confidence: float
    priority: str
    rationale: str


class BugInsights(BaseModel):
    severity: Optional[str] = None
    steps_to_reproduce: Optional[str] = None
    environment: Optional[str] = None
    impact_summary: Optional[str] = None


class FeatureInsights(BaseModel):
    user_value: Optional[str] = None
    demand_level: Optional[str] = None
    suggested_solution: Optional[str] = None

class TicketData(BaseModel):
    ticket_id: str
    title: str
    description: str
    category: str
    priority: str
    source_id: str
    source_type: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class QualityAssessment(BaseModel):
    passed: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: Optional[str] = None

class PipelineLog(BaseModel):
    source_id: str
    step: str
    status: str
    details: str


class PipelineMetrics(BaseModel):
    processed: int
    by_category: Dict[str, int]
    by_priority: Dict[str, int]
    avg_confidence: float

class GraphState(TypedDict, total=False):
    record: FeedbackRecord
    classification: ClassificationOutcome
    bug_insights: BugInsights
    feature_insights: FeatureInsights
    ticket: TicketData
    quality: QualityAssessment
    logs: List[PipelineLog]

@dataclass
class PipelineResult:
    ticket: TicketData
    quality: QualityAssessment
    logs: List[PipelineLog]
    classification: ClassificationOutcome
    bug_insights: Optional[BugInsights] = None
    feature_insights: Optional[FeatureInsights] = None
