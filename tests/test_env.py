import unittest

from env.actions import ActionType
from env.lifeops_env import LifeOpsEnv


class TestLifeOpsEnv(unittest.TestCase):
    def test_reset_returns_structured_state(self) -> None:
        env = LifeOpsEnv(seed=1)
        obs = env.reset("s1_basic_conflict")
        self.assertIn("calendar", obs)
        self.assertIn("tasks", obs)
        self.assertIn("current_request", obs)
        self.assertIsInstance(obs["calendar"], list)
        self.assertIsInstance(obs["tasks"], list)

    def test_valid_actions_are_constrained_to_current_request(self) -> None:
        env = LifeOpsEnv(seed=1)
        obs = env.reset("s1_basic_conflict")
        req_id = obs["current_request"]["event_id"]
        valid = env.valid_actions()
        # All request-handling actions should reference the current request id.
        for a in valid:
            if a.action_type in {
                ActionType.accept_event,
                ActionType.reject_event,
                ActionType.reschedule_event,
                ActionType.propose_new_time,
            }:
                self.assertEqual(a.request_id, req_id)

    def test_accept_moves_request_into_calendar(self) -> None:
        env = LifeOpsEnv(seed=1)
        obs = env.reset("s2_travel_tight")
        req_id = obs["current_request"]["event_id"]
        accept = next(a for a in env.valid_actions() if a.action_type == ActionType.accept_event)
        obs2, reward, done, info = env.step(accept)
        self.assertEqual(obs2["pending_request_count"], 0)
        self.assertTrue(any(e["event_id"] == req_id for e in obs2["calendar"]))

    def test_propose_does_not_add_to_calendar(self) -> None:
        env = LifeOpsEnv(seed=1)
        obs = env.reset("s2_travel_tight")
        req_id = obs["current_request"]["event_id"]
        propose = next(a for a in env.valid_actions() if a.action_type == ActionType.propose_new_time)
        obs2, reward, done, info = env.step(propose)
        self.assertEqual(obs2["pending_request_count"], 0)
        self.assertFalse(any(e["event_id"] == req_id for e in obs2["calendar"]))

    def test_accept_overlap_is_penalized(self) -> None:
        env = LifeOpsEnv(seed=1)
        env.reset("s1_basic_conflict")  # request overlaps standup
        accept = next(a for a in env.valid_actions() if a.action_type == ActionType.accept_event)
        obs2, reward, done, info = env.step(accept)
        self.assertLess(reward, 0.0)
        self.assertTrue(info["overlaps"])

    def test_travel_infeasible_is_penalized(self) -> None:
        env = LifeOpsEnv(seed=1)
        env.reset("s2_travel_tight")  # Downtown -> Office with 5 minute gap
        accept = next(a for a in env.valid_actions() if a.action_type == ActionType.accept_event)
        obs2, reward, done, info = env.step(accept)
        self.assertLess(reward, 0.0)
        self.assertTrue(info["travel_issues"])

    def test_block_focus_time_reduces_task_minutes(self) -> None:
        env = LifeOpsEnv(seed=1)
        obs = env.reset("s3_focus_vs_meeting")
        before = obs["tasks"][0]["remaining_minutes"]
        focus = next(a for a in env.valid_actions() if a.action_type == ActionType.block_focus_time)
        obs2, reward, done, info = env.step(focus)
        after = obs2["tasks"][0]["remaining_minutes"]
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()

