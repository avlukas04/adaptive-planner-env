"""
Tabular Q-learning agent for LifeOps (hackathon MVP).

Key design choices (per requirements):
- Q-values stored as nested dicts: Q[state_key][action] -> float
- Very simple state hash: state_key = str(state)
- Actions come from env.valid_actions(); we never select invalid actions.

This is intentionally minimal and framework-free.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple


ActionKey = Hashable  # we store action keys (tuples) as dict keys


def _state_key(state: Dict[str, Any]) -> str:
    """
    Requirement: state_key = str(state).

    Note: this is simple but can be large; acceptable for a hackathon MVP.
    """

    return str(state)


def _action_key(action: Any) -> ActionKey:
    """
    Make a stable, hashable key from an action object.

    LifeOps `Action` supports `.key()` which returns a tuple; we use that.
    If a raw dict is passed, we fall back to a tuple of common fields.
    """

    if hasattr(action, "key") and callable(getattr(action, "key")):
        return action.key()
    if isinstance(action, dict):
        return (
            action.get("action_type"),
            action.get("request_id"),
            action.get("new_start_min"),
            action.get("new_end_min"),
            action.get("duration_min"),
        )
    # Last resort: try hashing the object directly.
    return action


@dataclass
class QLearningAgent:
    """
    Lightweight epsilon-greedy Q-learning agent.
    """

    epsilon: float = 0.2
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    seed: Optional[int] = None

    q_table: Dict[str, Dict[ActionKey, float]] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0.0 <= self.discount_factor <= 1.0):
            raise ValueError("discount_factor must be in [0, 1]")

    def get_q(self, state_key: str, action_key: ActionKey) -> float:
        return float(self.q_table.get(state_key, {}).get(action_key, 0.0))

    def _ensure_state_row(self, state_key: str) -> Dict[ActionKey, float]:
        row = self.q_table.get(state_key)
        if row is None:
            row = {}
            self.q_table[state_key] = row
        return row

    def select_action(self, state: Dict[str, Any], valid_actions: Sequence[Any]) -> Any:
        """
        Epsilon-greedy action selection.

        Safety: only selects from `valid_actions`.
        """

        if not valid_actions:
            raise ValueError("valid_actions is empty; cannot select an action")

        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(valid_actions))

        s_key = _state_key(state)
        best_value: Optional[float] = None
        best_actions: List[Any] = []

        for a in valid_actions:
            ak = _action_key(a)
            q = self.get_q(s_key, ak)
            if best_value is None or q > best_value:
                best_value = q
                best_actions = [a]
            elif q == best_value:
                best_actions.append(a)

        # Tie-break randomly to avoid deterministic lock-in.
        return self._rng.choice(best_actions) if best_actions else self._rng.choice(list(valid_actions))

    def update(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Dict[str, Any],
        next_valid_actions: Sequence[Any],
        done: bool,
    ) -> None:
        """
        Standard tabular Q-learning update:

        Q(s,a) <- Q(s,a) + lr * (reward + gamma * max_a' Q(s',a') - Q(s,a))

        The prompt included a simplified formula; we implement the standard one.
        """

        s_key = _state_key(state)
        a_key = _action_key(action)
        sp_key = _state_key(next_state)

        current_q = self.get_q(s_key, a_key)

        if done or not next_valid_actions:
            target = float(reward)
        else:
            max_next = 0.0
            # Compute max_a' Q(s', a') over next state's valid actions.
            # Default Q is 0, so we start from 0.0 (optimistic-neutral).
            for a2 in next_valid_actions:
                q2 = self.get_q(sp_key, _action_key(a2))
                if q2 > max_next:
                    max_next = q2
            target = float(reward) + self.discount_factor * max_next

        new_q = current_q + self.learning_rate * (target - current_q)
        self._ensure_state_row(s_key)[a_key] = float(new_q)

