"""LangGraph pipeline (R3, KTD4): a thin adapter over the shared tool layer.

Four nodes — categorize, route, escalate, draft — on a LangGraph 1.x
``StateGraph``; escalation never suppresses the draft (R6, AE2). See
:mod:`triage.pipeline.graph` for wiring and :mod:`triage.pipeline.state`
for the state shape.
"""

from __future__ import annotations

from triage.pipeline.graph import build_graph, result_dict, run_pipeline
from triage.pipeline.state import PipelineState

__all__ = ["PipelineState", "build_graph", "result_dict", "run_pipeline"]
