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

