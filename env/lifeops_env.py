"""
LifeOps Environment (MVP).

This is a lightweight RL-style environment:
- structured JSON-like state (dicts/lists of primitives)
- constrained action space (generated from state)
- reward calculation (see env/reward.py)
- step() logic to mutate calendar/tasks and handle incoming requests

No external RL frameworks are used.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    # Normal usage (tests / `python -m ...`) expects repo root on sys.path.
    from env.actions import Action, ActionType, generate_valid_actions
    from env.personas import Persona, get_personas
    from env.reward import compute_reward, detect_overlaps, travel_issues
    from env.scenario_generator import Scenario, get_scenario, list_scenario_ids, sample_scenarios
except ModuleNotFoundError:
    # Hackathon-friendly: allow `python env/lifeops_env.py` from repo root.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from env.actions import Action, ActionType, generate_valid_actions
    from env.personas import Persona, get_personas
    from env.reward import compute_reward, detect_overlaps, travel_issues
    from env.scenario_generator import Scenario, get_scenario, list_scenario_ids, sample_scenarios


def _persona_to_dict(p: Persona) -> Dict[str, Any]:
    return {
        "persona_id": p.persona_id,
        "name": p.name,
        "home_location": p.home_location,
        "primary_work_location": p.primary_work_location,
        "preferred_meeting_window": [int(p.preferred_meeting_window[0]), int(p.preferred_meeting_window[1])],
        "avoid_meetings_before_min": p.avoid_meetings_before_min,
        "avoid_meetings_after_min": p.avoid_meetings_after_min,
        "preference_weight": float(p.preference_weight),
        "travel_aversion_weight": float(p.travel_aversion_weight),
        "focus_time_weight": float(p.focus_time_weight),
    }


def _action_key(action_dict: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(action_dict.get("action_type")),
        action_dict.get("request_id"),
        action_dict.get("new_start_min"),
        action_dict.get("new_end_min"),
        action_dict.get("duration_min"),
    )


@dataclass
class LifeOpsState:
    scenario_id: str
    persona: Dict[str, Any]
    calendar: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    pending_requests: List[Dict[str, Any]]
    travel_times: Dict[str, Dict[str, int]]
    step_count: int
    max_steps: int

    # Ephemeral fields used for reward shaping/debugging.
    last_added_event: Optional[Dict[str, Any]] = None
    last_handled_request: Optional[Dict[str, Any]] = None
    last_task_progress_minutes: int = 0

    def current_request(self) -> Optional[Dict[str, Any]]:
        return self.pending_requests[0] if self.pending_requests else None

    def to_observation(self) -> Dict[str, Any]:
        """
        JSON-like observation suitable for RL algorithms.

        Note: `travel_times` is included because it is part of feasibility.
        """

        return {
            "scenario_id": self.scenario_id,
            "persona": copy.deepcopy(self.persona),
            "calendar": copy.deepcopy(self.calendar),
            "tasks": copy.deepcopy(self.tasks),
            "current_request": copy.deepcopy(self.current_request()),
            "pending_request_count": len(self.pending_requests),
            "travel_times": copy.deepcopy(self.travel_times),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }


class LifeOpsEnv:
    """
    A minimal environment for one-day schedule management.

    Episode flow:
    - `reset()` loads a scenario with N incoming requests.
    - Each step handles the current request (accept/reject/reschedule/propose) or
      optionally blocks focus time (progresses tasks).
    - Episode ends when all requests are handled and tasks are complete, or after
      `max_steps`.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._personas = get_personas()
        self._state: Optional[LifeOpsState] = None

    def reset(self, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        if scenario_id is None:
            scenario_id = self._rng.choice(list_scenario_ids())
        scenario = get_scenario(scenario_id)

        persona = self._personas[scenario.persona_id]
        persona_dict = _persona_to_dict(persona)

        calendar = [e.to_dict() for e in scenario.calendar]
        tasks = [t.to_dict() for t in scenario.tasks]
        pending = [r.to_dict() for r in scenario.incoming_requests]

        max_steps = max(5, len(pending) + 5)
        self._state = LifeOpsState(
            scenario_id=scenario.scenario_id,
            persona=persona_dict,
            calendar=calendar,
            tasks=tasks,
            pending_requests=pending,
            travel_times=copy.deepcopy(scenario.travel_times),
            step_count=0,
            max_steps=max_steps,
        )
        return self._state.to_observation()

    def observation(self) -> Dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Call reset() before observation()")
        return self._state.to_observation()

    def valid_actions(self) -> List[Action]:
        obs = self.observation()
        return generate_valid_actions(obs)

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        action_dict = action.to_dict() if isinstance(action, Action) else dict(action)

        # Constrain actions to the current state's valid action set.
        valid = self.valid_actions()
        valid_keys = {a.key() for a in valid}
        if _action_key(action_dict) not in valid_keys:
            raise ValueError(f"Invalid action for current state: {action_dict}")

        prev_obs = self._state.to_observation()

        # Clear ephemeral fields.
        self._state.last_added_event = None
        self._state.last_handled_request = None
        self._state.last_task_progress_minutes = 0

        at = str(action_dict.get("action_type"))
        if at in {ActionType.accept_event.value, ActionType.reject_event.value, ActionType.reschedule_event.value, ActionType.propose_new_time.value}:
            self._apply_request_action(action_dict)
        elif at == ActionType.block_focus_time.value:
            self._apply_focus_action(action_dict)
        else:
            raise ValueError(f"Unknown action_type: {at}")

        self._state.step_count += 1

        # Done condition: all requests handled + tasks complete, or step limit.
        done = self._is_done()

        next_obs = self._state.to_observation()

        # For reward, include ephemeral fields (shaping/debug).
        reward_state = dict(next_obs)
        reward_state["last_added_event"] = copy.deepcopy(self._state.last_added_event)
        reward_state["last_handled_request"] = copy.deepcopy(self._state.last_handled_request)
        reward_state["last_task_progress_minutes"] = int(self._state.last_task_progress_minutes)

        reward, breakdown = compute_reward(prev_obs, action_dict, reward_state)

        info: Dict[str, Any] = {
            "reward_breakdown": breakdown,
            "overlaps": detect_overlaps(next_obs.get("calendar", [])),
            "travel_issues": travel_issues(next_obs.get("calendar", []), next_obs.get("travel_times", {})),
        }
        return next_obs, float(reward), bool(done), info

    def _apply_request_action(self, action_dict: Dict[str, Any]) -> None:
        req = self._state.current_request()
        if req is None:
            raise RuntimeError("No current_request to handle")

        req_id = req["event_id"]
        if action_dict.get("request_id") != req_id:
            raise ValueError("Action request_id does not match current_request")

        at = str(action_dict["action_type"])
        if at == ActionType.reject_event.value:
            self._state.last_handled_request = copy.deepcopy(req)
            self._state.pending_requests.pop(0)
            return

        # Accept/reschedule => schedule (add to calendar).
        # Propose_new_time => suggest without scheduling (does NOT add to calendar).
        suggested = copy.deepcopy(req)
        if at in {ActionType.reschedule_event.value, ActionType.propose_new_time.value}:
            ns = int(action_dict["new_start_min"])
            ne = int(action_dict["new_end_min"])
            if not (0 <= ns < ne <= 1440):
                raise ValueError("Invalid new_start_min/new_end_min")
            suggested["start_min"] = ns
            suggested["end_min"] = ne

        if at in {ActionType.accept_event.value, ActionType.reschedule_event.value}:
            self._state.calendar.append(suggested)

        # Record for reward shaping (preference handling) and debugging.
        self._state.last_added_event = copy.deepcopy(suggested)
        self._state.last_handled_request = copy.deepcopy(req)
        self._state.pending_requests.pop(0)

    def _apply_focus_action(self, action_dict: Dict[str, Any]) -> None:
        start = int(action_dict["new_start_min"])
        duration = int(action_dict["duration_min"])
        end = min(1440, start + duration)
        if not (0 <= start < end <= 1440):
            raise ValueError("Invalid focus block time")

        focus_event = {
            "event_id": f"focus_{self._state.step_count}",
            "title": "Focus time",
            "start_min": start,
            "end_min": end,
            "location": self._state.persona.get("primary_work_location", "Home"),
            "importance": 1,
            "kind": "focus",
        }
        self._state.calendar.append(focus_event)
        self._state.last_added_event = copy.deepcopy(focus_event)

        # Apply progress to the highest-priority unfinished task.
        unfinished = [t for t in self._state.tasks if int(t.get("remaining_minutes", 0)) > 0]
        unfinished.sort(key=lambda t: (-int(t.get("priority", 2)), str(t.get("task_id"))))
        if unfinished:
            t = unfinished[0]
            progress = min(duration, int(t["remaining_minutes"]))
            t["remaining_minutes"] = int(t["remaining_minutes"]) - progress
            self._state.last_task_progress_minutes = int(progress)
        else:
            self._state.last_task_progress_minutes = 0

    def _is_done(self) -> bool:
        if self._state.step_count >= self._state.max_steps:
            return True
        if self._state.pending_requests:
            return False
        if any(int(t.get("remaining_minutes", 0)) > 0 for t in self._state.tasks):
            return False
        return True


def _choose_simple_action(env: LifeOpsEnv) -> Action:
    """
    Tiny heuristic policy for manual running:
    - If accept would cause overlap/travel issues, try reschedule actions next.
    - Otherwise accept the request.
    - If no request, block focus time.
    """

    valid = env.valid_actions()
    obs = env.observation()
    req = obs.get("current_request")
    if req is None:
        focus_actions = [a for a in valid if a.action_type == ActionType.block_focus_time]
        if not focus_actions:
            return valid[0]

        def focus_score(a: Action) -> Tuple[int, int]:
            start = int(a.new_start_min or 0)
            dur = int(a.duration_min or 0)
            sim = list(obs.get("calendar", [])) + [
                {
                    "event_id": "focus_sim",
                    "start_min": start,
                    "end_min": start + dur,
                    "location": obs["persona"].get("primary_work_location", "Home"),
                }
            ]
            return (len(detect_overlaps(sim)), len(travel_issues(sim, obs.get("travel_times", {}))))

        focus_actions.sort(key=focus_score)
        return focus_actions[0]

    # Pick the request-handling action that minimizes feasibility violations.
    def score_action(a: Action) -> Tuple[int, int]:
        # (overlap_count, travel_issue_count) — smaller is better
        if a.action_type == ActionType.reject_event:
            return (999, 999)  # prefer scheduling over rejecting (manual runner)
        if a.action_type in {ActionType.accept_event, ActionType.reschedule_event, ActionType.propose_new_time}:
            added = dict(req)
            if a.action_type in {ActionType.reschedule_event, ActionType.propose_new_time}:
                added["start_min"] = int(a.new_start_min or added["start_min"])
                added["end_min"] = int(a.new_end_min or added["end_min"])
            # NOTE: propose_new_time does not actually schedule; don't add it to sim calendar.
            sim_events = list(obs.get("calendar", [])) + ([] if a.action_type == ActionType.propose_new_time else [added])
            return (len(detect_overlaps(sim_events)), len(travel_issues(sim_events, obs.get("travel_times", {}))))
        return (500, 500)

    candidates = [a for a in valid if a.action_type in {ActionType.accept_event, ActionType.reschedule_event, ActionType.propose_new_time, ActionType.reject_event}]
    candidates.sort(key=score_action)
    return candidates[0] if candidates else valid[0]


if __name__ == "__main__":
    # Simple manual episode runner: `python env/lifeops_env.py`
    env = LifeOpsEnv(seed=7)
    obs = env.reset()
    print("Scenario:", obs["scenario_id"])
    print("Persona:", obs["persona"]["name"])

    done = False
    total_reward = 0.0
    while not done:
        obs = env.observation()
        req = obs.get("current_request")
        if req is not None:
            print(f"\nCurrent request: {req['title']} ({req['start_min']}..{req['end_min']}) @ {req['location']}")
        else:
            print("\nNo pending requests.")

        action = _choose_simple_action(env)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        print("Action:", action.to_dict())
        print("Reward:", reward)
        if info.get("overlaps"):
            print("Overlaps:", info["overlaps"])
        if info.get("travel_issues"):
            print("Travel issues:", info["travel_issues"])

    print("\nEpisode done. Total reward:", total_reward)

