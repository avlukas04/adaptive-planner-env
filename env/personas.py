"""
Personas for LifeOps.

The environment is intentionally simple: personas encode a few preferences and
"soft constraints" used by the reward function (not hard rules).

All times are expressed as minutes since start of day (0..1440).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def h2m(hour: int, minute: int = 0) -> int:
    """Convert hour/minute to minutes since start of day."""
    if not (0 <= hour <= 24 and 0 <= minute < 60):
        raise ValueError("Invalid time")
    return hour * 60 + minute


@dataclass(frozen=True)
class Persona:
    """
    A lightweight user model.

    Notes:
    - These are preferences, not hard constraints. The agent can violate them,
      but will typically be penalized.
    - `preferred_meeting_window` is the "good" window for meetings/requests.
    """

    persona_id: str
    name: str
    home_location: str
    primary_work_location: str

    preferred_meeting_window: Tuple[int, int]
    avoid_meetings_before_min: Optional[int] = None
    avoid_meetings_after_min: Optional[int] = None

    # Relative weighting for reward shaping (hackathon-friendly knobs).
    preference_weight: float = 1.0
    travel_aversion_weight: float = 1.0
    focus_time_weight: float = 1.0

    def meeting_window_penalty(self, start_min: int, end_min: int) -> float:
        """
        Returns a non-negative penalty based on meeting time vs preferences.
        """

        penalty = 0.0
        pref_start, pref_end = self.preferred_meeting_window
        if start_min < pref_start or end_min > pref_end:
            penalty += 1.0
        if self.avoid_meetings_before_min is not None and start_min < self.avoid_meetings_before_min:
            penalty += 1.0
        if self.avoid_meetings_after_min is not None and end_min > self.avoid_meetings_after_min:
            penalty += 1.0
        return penalty * self.preference_weight


def get_personas() -> Dict[str, Persona]:
    """
    Returns a small fixed persona set for the MVP.

    Requirement: at least 3 personas.
    """

    return {
        "early_bird_engineer": Persona(
            persona_id="early_bird_engineer",
            name="Early-bird Engineer",
            home_location="Home",
            primary_work_location="Office",
            preferred_meeting_window=(h2m(9, 0), h2m(16, 30)),
            avoid_meetings_after_min=h2m(17, 30),
            preference_weight=1.2,
            travel_aversion_weight=1.0,
            focus_time_weight=1.1,
        ),
        "night_owl_creator": Persona(
            persona_id="night_owl_creator",
            name="Night-owl Creator",
            home_location="Home",
            primary_work_location="Downtown",
            preferred_meeting_window=(h2m(11, 0), h2m(19, 0)),
            avoid_meetings_before_min=h2m(10, 0),
            preference_weight=1.2,
            travel_aversion_weight=0.8,
            focus_time_weight=1.0,
        ),
        "busy_parent": Persona(
            persona_id="busy_parent",
            name="Busy Parent",
            home_location="Home",
            primary_work_location="Office",
            preferred_meeting_window=(h2m(9, 30), h2m(15, 0)),
            avoid_meetings_before_min=h2m(8, 30),
            avoid_meetings_after_min=h2m(16, 0),
            preference_weight=1.4,
            travel_aversion_weight=1.2,
            focus_time_weight=0.9,
        ),
    }

