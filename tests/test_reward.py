import unittest

from env.reward import detect_overlaps, travel_issues
from env.scenario_generator import default_travel_times


class TestRewardHelpers(unittest.TestCase):
    def test_detect_overlaps_finds_pair(self) -> None:
        events = [
            {"event_id": "a", "start_min": 60, "end_min": 120, "location": "Home"},
            {"event_id": "b", "start_min": 90, "end_min": 150, "location": "Home"},
        ]
        pairs = detect_overlaps(events)
        self.assertIn(("a", "b"), pairs)

    def test_travel_issues_flags_impossible_gap(self) -> None:
        tt = default_travel_times()
        events = [
            {"event_id": "a", "start_min": 600, "end_min": 660, "location": "Home"},
            # Home -> Downtown is 35 minutes in default matrix; gap here is 5.
            {"event_id": "b", "start_min": 665, "end_min": 700, "location": "Downtown"},
        ]
        issues = travel_issues(events, tt)
        self.assertTrue(issues)
        self.assertEqual(issues[0][0], "a")
        self.assertEqual(issues[0][1], "b")


if __name__ == "__main__":
    unittest.main()




class TestNewRewardTerms:
    def test_important_accept_bonus(self):
        from env.reward import compute_reward
        from env.actions import ActionType

        prev_state = {
            "calendar": [],
            "persona": {"travel_aversion_weight": 1.0, "home_location": "Office"},
            "travel_times": {},
        }
        next_state = {
            "calendar": [{"start_min": 600, "end_min": 660, "location": "Office", "title": "Big Meeting"}],
            "persona": {"travel_aversion_weight": 1.0, "home_location": "Office"},
            "travel_times": {},
            "last_handled_request": {"importance": 3, "title": "Big Meeting"},
        }
        action = {"action_type": "accept_event", "request_id": "r1"}

        reward, breakdown = compute_reward(prev_state, action, next_state)
        assert "important_accept_bonus" in breakdown, f"Got: {breakdown}"
        assert breakdown["important_accept_bonus"] == 2.0

    def test_clean_schedule_bonus(self):
        from env.reward import compute_reward

        prev_state = {
            "calendar": [],
            "persona": {"travel_aversion_weight": 1.0, "home_location": "Office"},
            "travel_times": {},
        }
        next_state = {
            "calendar": [{"start_min": 600, "end_min": 660, "location": "Office", "title": "Lunch"}],
            "persona": {"travel_aversion_weight": 1.0, "home_location": "Office"},
            "travel_times": {},
            "last_handled_request": {"importance": 1, "title": "Lunch"},
        }
        action = {"action_type": "accept_event", "request_id": "r1"}

        reward, breakdown = compute_reward(prev_state, action, next_state)
        assert "clean_schedule_bonus" in breakdown, f"Got: {breakdown}"
        assert breakdown["clean_schedule_bonus"] == 1.0
