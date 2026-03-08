"""
OpenEnv models for LifeOps - Pydantic Action and Observation.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class LifeOpsAction(Action):
    """Action for LifeOps: accept/reject/reschedule/propose/block_focus_time."""

    action_type: str = Field(..., description="One of: accept_event, reject_event, reschedule_event, propose_new_time, block_focus_time")
    request_id: Optional[str] = Field(default=None, description="Required for request-handling actions")
    new_start_min: Optional[int] = Field(default=None, ge=0, le=1440)
    new_end_min: Optional[int] = Field(default=None, ge=0, le=1440)
    duration_min: Optional[int] = Field(default=None, ge=0, le=480)


class LifeOpsObservation(Observation):
    """Observation from LifeOps: calendar state, persona, valid actions."""

    observation: Dict[str, Any] = Field(..., description="Full observation: scenario_id, persona, calendar, tasks, current_request, etc.")
    valid_actions: List[Dict[str, Any]] = Field(default_factory=list, description="List of valid actions as dicts")
