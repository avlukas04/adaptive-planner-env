#!/usr/bin/env python3
"""
Minimal RL training loop for LifeOps.

Runs episodes in the environment, collects trajectories, and prints results.
Uses a simple policy (random or heuristic). No external RL frameworks required.

For learned policies, consider adding HuggingFace TRL or a small PyTorch policy.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root for imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.actions import Action
from env.lifeops_env import LifeOpsEnv, _choose_simple_action


def random_policy(env: LifeOpsEnv) -> Action:
    """Pick uniformly from valid actions."""
    valid = env.valid_actions()
    if not valid:
        raise RuntimeError("No valid actions")
    return random.choice(valid)


def collect_trajectory(
    env: LifeOpsEnv,
    policy: str = "random",
    scenario_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], float, int, str]:
    """
    Run one episode and collect trajectory.

    Returns:
        trajectory: list of (obs, action_dict, reward, done) per step
        total_reward: sum of rewards
        episode_length: number of steps
        scenario_id: scenario used
    """
    obs = env.reset(scenario_id=scenario_id)
    scenario_id = obs["scenario_id"]
    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_count = 0

    policy_fn = _choose_simple_action if policy == "heuristic" else random_policy

    done = False
    while not done:
        action = policy_fn(env)
        action_dict = action.to_dict()

        next_obs, reward, done, info = env.step(action)
        step_count += 1
        total_reward += reward

        trajectory.append({
            "obs": obs,
            "action": action_dict,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "info": info,
        })
        obs = next_obs

    return trajectory, total_reward, step_count, scenario_id


def _format_action_short(a: Dict[str, Any]) -> str:
    """Format action for key decisions summary."""
    at = a.get("action_type", "?")
    if at == "block_focus_time":
        start = a.get("new_start_min", 0)
        dur = a.get("duration_min", 0)
        h, m = divmod(start or 0, 60)
        return f"block_focus @ {h:02d}:{m:02d} ({dur}min)"
    if at == "accept_event":
        return f"accept request {a.get('request_id', '?')}"
    if at == "reject_event":
        return f"reject request {a.get('request_id', '?')}"
    if at == "reschedule_event":
        ns = a.get("new_start_min", 0)
        h, m = divmod(ns or 0, 60)
        return f"reschedule → {h:02d}:{m:02d}"
    if at == "propose_new_time":
        ns = a.get("new_start_min", 0)
        h, m = divmod(ns or 0, 60)
        return f"propose → {h:02d}:{m:02d}"
    return at


def print_episode_results(
    episode: int,
    total_reward: float,
    episode_length: int,
    scenario_id: str,
    trajectory: List[Dict[str, Any]],
    verbose: bool = False,
) -> None:
    """Print human-readable episode results."""
    print(f"\n--- Episode {episode} ---")
    print(f"  Scenario:      {scenario_id}")
    print(f"  Steps:         {episode_length}")
    print(f"  Total reward:  {total_reward:+.2f}")

    # Key decisions taken
    if trajectory:
        decisions = [_format_action_short(t["action"]) for t in trajectory]
        print(f"  Key decisions: {', '.join(decisions)}")

    if verbose:
        for i, t in enumerate(trajectory):
            a = t["action"]
            at = a.get("action_type", "?")
            r = t["reward"]
            print(f"    Step {i + 1}: {at}  reward={r:+.2f}")


def train(
    num_episodes: int = 20,
    seed: Optional[int] = 42,
    policy: str = "random",
    scenario_id: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run RL training loop: collect trajectories and print results.

    Args:
        num_episodes: number of episodes to run
        seed: random seed for env (None = random)
        policy: "random" or "heuristic"
        scenario_id: fix scenario (None = random each episode)
        verbose: print per-step details

    Returns:
        Summary dict with episode rewards and stats
    """
    env = LifeOpsEnv(seed=seed)

    all_rewards: List[float] = []
    all_lengths: List[int] = []
    all_scenarios: List[str] = []

    print("=" * 50)
    print("LifeOps RL Training")
    print("=" * 50)
    print(f"Episodes: {num_episodes}  |  Policy: {policy}  |  Seed: {seed}")

    for ep in range(1, num_episodes + 1):
        trajectory, total_reward, ep_len, scenario_id_used = collect_trajectory(
            env, policy=policy, scenario_id=scenario_id
        )

        all_rewards.append(total_reward)
        all_lengths.append(ep_len)
        all_scenarios.append(scenario_id_used)

        print_episode_results(
            episode=ep,
            total_reward=total_reward,
            episode_length=ep_len,
            scenario_id=scenario_id_used,
            trajectory=trajectory,
            verbose=verbose,
        )

    # Summary
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_len = sum(all_lengths) / len(all_lengths)
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"  Episodes:     {num_episodes}")
    print(f"  Avg reward:   {avg_reward:+.2f}")
    print(f"  Avg length:   {avg_len:.1f} steps")
    print(f"  Best reward:  {max(all_rewards):+.2f}")
    print(f"  Worst reward: {min(all_rewards):+.2f}")
    print("=" * 50)

    return {
        "rewards": all_rewards,
        "lengths": all_lengths,
        "scenarios": all_scenarios,
        "avg_reward": avg_reward,
        "avg_length": avg_len,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent on LifeOps")
    parser.add_argument("-n", "--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("-p", "--policy", choices=["random", "heuristic"], default="random")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None, help="Fix scenario (e.g. s1_basic_conflict)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        seed=args.seed,
        policy=args.policy,
        scenario_id=args.scenario,
        verbose=args.verbose,
    )
