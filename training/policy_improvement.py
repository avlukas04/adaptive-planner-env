"""
Policy improvement techniques for LifeOps RL training.

Implements reward-based methods that work with LLM policies (no gradients needed):

1. Best-of-N: At each step, sample N actions from the LLM, simulate each for 1 step,
   pick the action with highest immediate reward. Improves over greedy by exploring.

2. In-context learning: Maintain a replay buffer of (state, action, reward) from
   high-reward trajectories. Prepend 1-2 examples to the prompt as few-shot demos.
   The prompt improves over time as we collect better examples.

3. Best trajectory replay: Run N full episodes, keep the best trajectory, use its
   decisions as few-shot examples for subsequent episodes.

Usage:
    from training.policy_improvement import best_of_n_policy_fn, in_context_policy_fn
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _focus_overlaps_calendar(action: Any, calendar: List[Dict[str, Any]]) -> bool:
    """True if adding this focus block would overlap with existing calendar events."""
    from env.reward import detect_overlaps
    start = int(action.new_start_min or 0)
    dur = int(action.duration_min or 0)
    sim = list(calendar) + [
        {"event_id": "_", "start_min": start, "end_min": start + dur, "location": "x"},
    ]
    return len(detect_overlaps(sim)) > 0


def filter_overlapping_focus_actions(
    valid_actions: List[Any],
    calendar: List[Dict[str, Any]],
) -> List[Any]:
    """
    Remove focus block actions that would overlap with the current calendar.
    Matches baseline heuristic behavior. If all focus blocks overlap, keep all
    (fallback to least-bad).
    """
    from env.actions import ActionType
    focus_actions = [a for a in valid_actions if a.action_type == ActionType.block_focus_time]
    other_actions = [a for a in valid_actions if a.action_type != ActionType.block_focus_time]
    if not focus_actions:
        return valid_actions
    non_overlapping = [a for a in focus_actions if not _focus_overlaps_calendar(a, calendar)]
    candidates = non_overlapping if non_overlapping else focus_actions
    return other_actions + candidates


@dataclass(order=True)
class _TrajectoryEntry:
    """Entry for replay buffer: (negative reward for min-heap, trajectory)."""
    reward: float
    trajectory: List[Dict[str, Any]] = field(compare=False)

    def __post_init__(self):
        # Heapq is a min-heap; we want highest reward first, so store -reward
        object.__setattr__(self, "reward", -self.reward)


def best_of_n_policy_fn(
    env: Any,
    llm_choose_samples: Callable[[Dict, List, Optional[str], int, float, Callable], List[Any]],
    fallback_fn: Callable[[], Any],
    model_id: Optional[str] = None,
    n: int = 5,
    temperature: float = 0.7,
) -> Any:
    """
    Best-of-N policy: sample N actions, simulate each 1 step, pick the one with
    highest immediate reward.

    Args:
        env: LifeOpsEnv (must have clone() and step())
        llm_choose_samples: choose_action_samples from agent.llm_agent
        fallback_fn: e.g. lambda: _choose_simple_action(env)
        model_id: LLM model ID
        n: number of samples per step
        temperature: sampling temperature (0.7 typical for diversity)
    """
    state = env.observation()
    valid_actions = env.valid_actions()
    # Filter out focus blocks that would overlap (matches baseline heuristic)
    calendar = state.get("calendar", [])
    valid_actions = filter_overlapping_focus_actions(valid_actions, calendar)
    if not valid_actions:
        raise RuntimeError("No valid actions")

    samples = llm_choose_samples(
        state,
        valid_actions,
        model_id=model_id,
        num_samples=n,
        temperature=temperature,
        fallback_fn=fallback_fn,
    )

    # Deduplicate by action key; for each unique action, simulate 1 step
    seen_keys: set = set()
    candidates: List[Tuple[float, Any]] = []

    for action in samples:
        key = action.key()
        if key in seen_keys:
            continue
        seen_keys.add(key)

        cloned = env.clone()
        try:
            _, reward, _, _ = cloned.step(action)
            candidates.append((reward, action))
        except Exception as e:
            logger.debug("Simulation failed for action %s: %s", action, e)

    if not candidates:
        return fallback_fn()

    # Pick action with highest immediate reward
    best_reward, best_action = max(candidates, key=lambda x: x[0])
    return best_action


def _min_to_time(m: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


def _format_example_for_prompt(obs: Dict, action_dict: Dict, reward: float) -> str:
    """Format one (obs, action, reward) as a few-shot example."""
    lines = []
    req = obs.get("current_request")
    if req:
        lines.append(f"Request: {req.get('title', '?')} @ {_min_to_time(int(req.get('start_min', 0)))}")
    at = action_dict.get("action_type", "?")
    if at == "block_focus_time":
        start = action_dict.get("new_start_min", 0)
        dur = action_dict.get("duration_min", 0)
        lines.append(f"Chose: block_focus_time @ {_min_to_time(int(start or 0))} for {dur} min")
    elif at == "reject_event":
        lines.append(f"Chose: reject_event (request {action_dict.get('request_id', '?')})")
    elif at == "accept_event":
        lines.append(f"Chose: accept_event (request {action_dict.get('request_id', '?')})")
    elif at in ("reschedule_event", "propose_new_time"):
        ns = action_dict.get("new_start_min", 0)
        lines.append(f"Chose: {at} → {_min_to_time(int(ns or 0))}")
    else:
        lines.append(f"Chose: {at}")
    lines.append(f"Reward: {reward:+.2f}")
    return " | ".join(lines)


class InContextReplayBuffer:
    """
    Replay buffer of best trajectories for in-context learning.
    Keeps the top K trajectories by total reward.
    """

    def __init__(self, max_size: int = 5, min_reward: float = 0.0):
        self.max_size = max_size
        self.min_reward = min_reward
        self._heap: List[_TrajectoryEntry] = []

    def add(self, trajectory: List[Dict[str, Any]], total_reward: float) -> None:
        if total_reward < self.min_reward:
            return
        entry = _TrajectoryEntry(reward=total_reward, trajectory=trajectory)
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, entry)
        elif total_reward > -self._heap[0].reward:
            heapq.heapreplace(self._heap, entry)

    def get_best_examples(self, num_steps: int = 2) -> List[str]:
        """
        Return formatted few-shot examples from the best trajectories.
        Takes up to num_steps from the best trajectory (first steps are usually
        most informative).
        """
        if not self._heap:
            return []
        # Best is the one with smallest (most negative) -reward, i.e. highest reward
        best = max(self._heap, key=lambda e: -e.reward)
        examples = []
        for t in best.trajectory[:num_steps]:
            ex = _format_example_for_prompt(
                t["obs"], t["action"], t["reward"]
            )
            examples.append(ex)
        return examples


def in_context_policy_fn(
    env: Any,
    choose_with_context: Callable,
    fallback_fn: Callable[[], Any],
    parse_stats: Optional[Dict[str, int]] = None,
) -> Any:
    """
    LLM policy augmented with few-shot examples from best past trajectories.
    Use create_in_context_choose_action() to build choose_with_context.
    """
    state = env.observation()
    valid_actions = env.valid_actions()
    return choose_with_context(
        state,
        valid_actions,
        fallback_fn=fallback_fn,
        parse_stats=parse_stats,
    )


def create_in_context_choose_action(
    replay_buffer: InContextReplayBuffer,
    num_examples: int = 2,
    model_id: Optional[str] = None,
) -> Callable:
    """
    Create a choose_action wrapper that prepends few-shot examples to the prompt.
    """
    def choose_with_context(
        state: Dict[str, Any],
        valid_actions: List[Any],
        model_id_override: Optional[str] = None,
        fallback_fn: Optional[Callable[[], Any]] = None,
        parse_stats: Optional[Dict[str, int]] = None,
    ) -> Any:
        from agent.llm_agent import choose_action
        examples = replay_buffer.get_best_examples(num_steps=num_examples)
        prefix = ""
        if examples:
            prefix = "Examples of good decisions (state → action → reward):\n"
            for ex in examples:
                prefix += f"  {ex}\n"
            prefix += "\n"
        return choose_action(
            state, valid_actions,
            model_id=model_id_override or model_id,
            fallback_fn=fallback_fn,
            parse_stats=parse_stats,
            few_shot_prefix=prefix if prefix else None,
        )
    return choose_with_context
