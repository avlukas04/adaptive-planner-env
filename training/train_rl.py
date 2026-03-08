#!/usr/bin/env python3
"""
Minimal RL training loop for LifeOps.

Runs episodes in the environment, collects trajectories, and prints results.
Uses a simple policy (random or heuristic). No external RL frameworks required.

For learned policies, consider adding HuggingFace TRL or a small PyTorch policy.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add repo root for imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.actions import Action, mask_illegal_actions
from env.baseline_agent import choose_baseline_action
from env.lifeops_env import LifeOpsEnv
from env.scenario_generator import edge_case_scenario_ids, scenario_ids_by_difficulty
from llm_agent import LLMAgent


def random_policy(env: LifeOpsEnv) -> Action:
    """Pick uniformly from valid actions."""
    state = env.observation()
    valid = mask_illegal_actions(state, env.valid_actions())
    if not valid:
        raise RuntimeError("No valid actions")
    return random.choice(valid)

def baseline_policy(env: LifeOpsEnv) -> Action:
    """Rule-based baseline floor policy."""
    state = env.observation()
    return choose_baseline_action(state, env.valid_actions())


def collect_trajectory(
    env: LifeOpsEnv,
    policy_fn: Callable[[LifeOpsEnv], Action],
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
    llm_agent: Optional[LLMAgent] = None
    if policy == "llm":
        llm_agent = LLMAgent(local_model_name=os.environ.get("LIFEOPS_LOCAL_MODEL"))  # type: ignore[name-defined]

    all_rewards: List[float] = []
    all_lengths: List[int] = []
    all_scenarios: List[str] = []

    print("=" * 50)
    print("LifeOps RL Training")
    print("=" * 50)
    print(f"Episodes: {num_episodes}  |  Policy: {policy}  |  Seed: {seed}")

    def ma10(vals: List[float]) -> float:
        w = 10
        tail = vals[-w:] if len(vals) >= 1 else []
        return (sum(tail) / float(len(tail))) if tail else 0.0

    for ep in range(1, num_episodes + 1):
        # Curriculum: easy -> medium -> hard (ramp every 20 episodes), unless user fixed a scenario.
        if scenario_id is None:
            if ep <= 20:
                diff = "easy"
            elif ep <= 40:
                diff = "medium"
            else:
                diff = "hard"
            pool = scenario_ids_by_difficulty(diff)
            sid = random.choice(pool) if pool else None
            # Edge cases ~30% of episodes.
            edge_ids = edge_case_scenario_ids()
            if edge_ids and random.random() < 0.30:
                sid = random.choice(edge_ids)
        else:
            diff = "fixed"
            sid = scenario_id

        if policy == "random":
            pf = random_policy
        elif policy == "baseline":
            pf = baseline_policy
        elif policy == "llm":
            assert llm_agent is not None
            pf = llm_agent.choose_action  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown policy: {policy}")

        trajectory, total_reward, ep_len, scenario_id_used = collect_trajectory(env, policy_fn=pf, scenario_id=sid)

        all_rewards.append(total_reward)
        all_lengths.append(ep_len)
        all_scenarios.append(scenario_id_used)

        # Constraint logging: which violations occurred most in this episode.
        overlap_steps = sum(1 for t in trajectory if t.get("info", {}).get("overlaps"))
        travel_steps = sum(1 for t in trajectory if t.get("info", {}).get("travel_issues"))
        pref_steps = sum(1 for t in trajectory if (t.get("info", {}).get("reward_breakdown") or {}).get("preference_penalty"))
        remaining = sum(int(t.get("remaining_minutes", 0)) for t in (trajectory[-1]["next_obs"].get("tasks", []) if trajectory else []))
        goal_miss = 1 if remaining > 0 else 0
        counts = {"travel": travel_steps, "double_booking": overlap_steps, "preference": pref_steps, "goal_miss": goal_miss}
        most = "none" if max(counts.values()) <= 0 else max(counts.items(), key=lambda kv: kv[1])[0]

        print(f"\nEpisode {ep}/{num_episodes} | diff={diff} | scenario={scenario_id_used}")
        print(f"  reward={total_reward:+.2f} | ma10={ma10(all_rewards):+.2f} | steps={ep_len} | most_violated={most}")
        print(f"  violations: travel={travel_steps} overlap={overlap_steps} preference={pref_steps} goal_miss={goal_miss}")

        print_episode_results(
            episode=ep,
            total_reward=total_reward,
            episode_length=ep_len,
            scenario_id=scenario_id_used,
            trajectory=trajectory,
            verbose=verbose,
        )

        # Update LLM memory after the episode.
        if llm_agent is not None:
            llm_agent.on_episode_end(trajectory, total_reward=float(total_reward))

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
    parser.add_argument("-p", "--policy", choices=["random", "baseline", "llm"], default="random")
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
