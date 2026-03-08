"""
LifeOps OpenEnv client - connects to LifeOps environment on HF Spaces.
"""

from typing import Any, Dict

from openenv_core.client_types import StepResult
from openenv_core.env_client import EnvClient
from openenv_core.env_server.types import State

from .models import LifeOpsAction, LifeOpsObservation


class LifeOpsEnv(EnvClient[LifeOpsAction, LifeOpsObservation, State]):
    """Client for the LifeOps environment on HF Spaces."""

    def _step_payload(self, action: LifeOpsAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "request_id": action.request_id,
            "new_start_min": action.new_start_min,
            "new_end_min": action.new_end_min,
            "duration_min": action.duration_min,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LifeOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = LifeOpsObservation(
            observation=obs_data.get("observation", obs_data),
            valid_actions=obs_data.get("valid_actions", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
