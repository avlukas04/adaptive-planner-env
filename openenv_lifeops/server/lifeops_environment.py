"""
OpenEnv server: LifeOpsEnvironment wraps LifeOpsEnv for OpenEnv 0.2.1.
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add repo root so we can import env
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from env.actions import Action, ActionType, generate_valid_actions
from env.lifeops_env import LifeOpsEnv

from openenv_core.env_server.interfaces import Environment
from openenv_core.env_server.types import State

from ..models import LifeOpsAction, LifeOpsObservation


def _action_to_dict(a: LifeOpsAction) -> dict:
    return {
        "action_type": a.action_type,
        "request_id": a.request_id,
        "new_start_min": a.new_start_min,
        "new_end_min": a.new_end_min,
        "duration_min": a.duration_min,
    }


def _dict_to_action(d: dict) -> Action:
    return Action(
        action_type=ActionType(d["action_type"]),
        request_id=d.get("request_id"),
        new_start_min=d.get("new_start_min"),
        new_end_min=d.get("new_end_min"),
        duration_min=d.get("duration_min"),
    )


class LifeOpsEnvironment(Environment[LifeOpsAction, LifeOpsObservation, State]):
    """LifeOps schedule management environment for OpenEnv."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: LifeOpsEnv | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario_id: str | None = None,
        **kwargs,
    ) -> LifeOpsObservation:
        self._env = LifeOpsEnv(seed=seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        obs_dict = self._env.reset(scenario_id=scenario_id)
        valid = [a.to_dict() for a in self._env.valid_actions()]
        return LifeOpsObservation(
            observation=obs_dict,
            valid_actions=valid,
            done=False,
            reward=None,
            metadata={"scenario_id": obs_dict.get("scenario_id")},
        )

    def step(
        self,
        action: LifeOpsAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> LifeOpsObservation:
        if self._env is None:
            raise RuntimeError("Call reset() before step()")

        action_dict = _action_to_dict(action)
        env_action = _dict_to_action(action_dict)

        next_obs, reward, done, info = self._env.step(env_action)
        self._state.step_count += 1

        valid = [a.to_dict() for a in self._env.valid_actions()]
        return LifeOpsObservation(
            observation=next_obs,
            valid_actions=valid,
            done=done,
            reward=float(reward),
            metadata={
                "reward_breakdown": info.get("reward_breakdown", {}),
                "overlaps": info.get("overlaps", []),
                "travel_issues": info.get("travel_issues", []),
            },
        )

    @property
    def state(self) -> State:
        return self._state
