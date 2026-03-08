"""
Adapter to use OpenEnv LifeOpsEnv client as a drop-in for local LifeOpsEnv.
Enables train_rl.collect_trajectory and Colab scripts to work with remote env.
"""

from typing import Any, Dict, List, Optional

from env.actions import Action, ActionType

from .client import LifeOpsEnv as OpenEnvLifeOpsClient
from .models import LifeOpsAction


def _dict_to_action(d: dict) -> Action:
    return Action(
        action_type=ActionType(d["action_type"]),
        request_id=d.get("request_id"),
        new_start_min=d.get("new_start_min"),
        new_end_min=d.get("new_end_min"),
        duration_min=d.get("duration_min"),
    )


def _action_to_lifeops_action(a: Action) -> LifeOpsAction:
    return LifeOpsAction(
        action_type=a.action_type.value,
        request_id=a.request_id,
        new_start_min=a.new_start_min,
        new_end_min=a.new_end_min,
        duration_min=a.duration_min,
    )


class LifeOpsEnvAdapter:
    """
    Wraps OpenEnv LifeOpsEnv client to match local LifeOpsEnv interface.
    Use with train_rl.collect_trajectory and agent policies.
    """

    def __init__(self, base_url: str, **client_kwargs):
        self._client = OpenEnvLifeOpsClient(base_url=base_url, **client_kwargs)
        self._client.__enter__()
        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_valid_actions: List[Action] = []

    def reset(self, scenario_id: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        result = self._client.reset(scenario_id=scenario_id, seed=seed)
        self._last_obs = result.observation.observation
        self._last_valid_actions = [_dict_to_action(d) for d in result.observation.valid_actions]
        return self._last_obs

    def valid_actions(self) -> List[Action]:
        return self._last_valid_actions

    def step(self, action: Action) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        lifeops_action = _action_to_lifeops_action(action)
        result = self._client.step(lifeops_action)
        self._last_obs = result.observation.observation
        self._last_valid_actions = [_dict_to_action(d) for d in result.observation.valid_actions]
        info = result.observation.metadata or {}
        return self._last_obs, float(result.reward or 0.0), bool(result.done), info

    def close(self):
        self._client.close()
