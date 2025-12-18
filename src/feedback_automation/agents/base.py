"""Shared abstractions and utilities for agent implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..config import ApplicationConfig
from ..schemas import GraphState
from ..utils import create_llm

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all processing agents.
    Handles configuration and optional LLM initialization.
    """

    def __init__(self, config: ApplicationConfig) -> None:
        self.config = config
        self.llm = create_llm(config)

    @abstractmethod
    def run(self, state: GraphState) -> GraphState:
        """
        Execute agent logic and return updated graph state.
        """
        raise NotImplementedError("Agents must implement the run() method")


def merge_payloads(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two dictionaries, with `overrides` taking precedence.
    """
    result = dict(base)
    result.update(overrides)
    return result
