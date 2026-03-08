#!/usr/bin/env python3
"""
Minimal RL training loop for LifeOps.

Runs episodes in the environment, collects trajectories, and prints results.
Uses a simple policy (random or heuristic). No external RL frameworks required.

Tracks: episode rewards, moving average reward, episode length.
Returns result["rewards"] and result["lengths"] for plotting, e.g.:
    result = train(num_episodes=100)
    import matplotlib.pyplot as plt
    plt.plot(result["rewards"])
    plt.show()

For learned policies, consider adding HuggingFace TRL or a small PyTorch policy.
"""

from __future__ import annotations

# Load .env before any HuggingFace imports (for HF_TOKEN)
import os as _os
_env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
if _os.path.isfile(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

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
from env.scenario_generator import list_scenario_ids

try:
    from agent.llm_agent import choose_action as llm_choose_action, choose_action_samples
except ImportError:
    llm_choose_action = None
    choose_action_samples = None

try:
    from training.policy_improvement import (
        best_of_n_policy_fn,
        create_in_context_choose_action,
        filter_overlapping_focus_actions,
        InContextReplayBuffer,
    )
except ImportError:
    best_of_n_policy_fn = None
    create_in_context_choose_action = None
    filter_overlapping_focus_actions = None
    InContextReplayBuffer = None


def random_policy(env: LifeOpsEnv) -> Action:
    """Pick uniformly from valid actions."""
    valid = env.valid_actions()
    if not valid:
        raise RuntimeError("No valid actions")
    return random.choice(valid)


def collect_trajectory(
    env: LifeOpsEnv,
    agent: str = "random",
    scenario_id: Optional[str] = None,
    llm_model_id: Optional[str] = None,
    parse_stats: Optional[Dict[str, int]] = None,
    llm_method: str = "vanilla",
    best_of_n: int = 5,
    in_context_choose_fn: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], float, int, str, str]:
    """
    Run one episode and collect trajectory.

    When scenario_id is None, randomly samples a scenario from scenario_generator.
    Each scenario has an associated persona (scenario determines persona).

    Returns:
        trajectory: list of (obs, action_dict, reward, done) per step
        total_reward: sum of rewards
        episode_length: number of steps
        scenario_id: scenario used
        persona_name: persona name (from scenario)
    """
    if scenario_id is None:
        scenario_id = env._rng.choice(list_scenario_ids())
    obs = env.reset(scenario_id=scenario_id)
    scenario_id = obs["scenario_id"]
    persona_name = obs["persona"].get("name", "?")
    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_count = 0

    if agent == "baseline":
        policy_fn = _choose_simple_action
    elif agent == "llm":
        if llm_method == "best-of-n" and best_of_n_policy_fn is not None and choose_action_samples is not None:
            policy_fn = lambda e: best_of_n_policy_fn(
                e,
                llm_choose_samples=choose_action_samples,
                fallback_fn=lambda: _choose_simple_action(env),
                model_id=llm_model_id,
                n=best_of_n,
                temperature=0.9,  # higher for more diverse samples
            )
        elif llm_method == "in-context" and in_context_choose_fn is not None:
            def _in_context_policy(e):
                obs = e.observation()
                valid = e.valid_actions()
                if filter_overlapping_focus_actions is not None:
                    valid = filter_overlapping_focus_actions(valid, obs.get("calendar", []))
                return in_context_choose_fn(
                    obs, valid,
                    fallback_fn=lambda: _choose_simple_action(env),
                    parse_stats=parse_stats,
                )
            policy_fn = _in_context_policy
        elif llm_choose_action is not None:
            def _vanilla_llm_policy(e):
                obs = e.observation()
                valid = e.valid_actions()
                if filter_overlapping_focus_actions is not None:
                    valid = filter_overlapping_focus_actions(valid, obs.get("calendar", []))
                return llm_choose_action(
                    obs, valid,
                    model_id=llm_model_id,
                    fallback_fn=lambda: _choose_simple_action(env),
                    parse_stats=parse_stats,
                )
            policy_fn = _vanilla_llm_policy
        else:
            policy_fn = random_policy  # fallback if agent not available
    else:
        policy_fn = random_policy

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

    return trajectory, total_reward, step_count, scenario_id, persona_name


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


def print_episode_summary(
    episode: int,
    total_reward: float,
    episode_length: int,
    scenario_id: str,
    persona_name: str,
    trajectory: List[Dict[str, Any]],
    moving_avg: float,
    moving_avg_window: int,
    verbose: bool = False,
) -> None:
    """Print a readable episode summary for hackathon demo."""
    print()
    print("  " + "=" * 50)
    print(f"  EPISODE {episode} SUMMARY")
    print("  " + "=" * 50)
    print(f"  Persona:   {persona_name}")
    print(f"  Scenario:  {scenario_id}")
    print(f"  Length:    {episode_length} steps")
    print(f"  Reward:    {total_reward:+.2f}")
    print(f"  Mov avg:   {moving_avg:+.2f} (last {moving_avg_window} ep)")
    print("  " + "-" * 50)
    print("  Action sequence:")
    if trajectory:
        action_seq = [_format_action_short(t["action"]) for t in trajectory]
        for i, (act, t) in enumerate(zip(action_seq, trajectory)):
            r = t["reward"]
            print(f"    {i + 1}. {act}  →  {r:+.2f}")
    else:
        print("    (none)")
    print("  " + "=" * 50)

    if verbose:
        print("  Per-step detail:")
        for i, t in enumerate(trajectory):
            a = t["action"]
            at = a.get("action_type", "?")
            r = t["reward"]
            print(f"    Step {i + 1}: {at}  →  reward {r:+.2f}")


def _moving_average(values: List[float], window: int) -> float:
    """Compute moving average over last `window` values."""
    if not values:
        return 0.0
    recent = values[-window:]
    return sum(recent) / len(recent)


def _plot_rewards(rewards: List[float], window: int = 10, policy: str = "") -> None:
    """Plot episode rewards and moving average. Skips if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed, skipping plot)")
        return

    episodes = list(range(1, len(rewards) + 1))
    ma = [
        sum(rewards[max(0, i - window) : i + 1]) / min(i + 1, window)
        for i in range(len(rewards))
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, "o-", alpha=0.5, markersize=4, label="Episode reward")
    plt.plot(episodes, ma, "-", linewidth=2, label=f"Moving avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"LifeOps Training{f' ({policy})' if policy else ''}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _print_periodic_summary(
    episode: int,
    rewards: List[float],
    lengths: List[int],
    window: int = 10,
) -> None:
    """Print moving average summary every N episodes."""
    ma_reward = _moving_average(rewards, window)
    recent_lengths = lengths[-window:]
    ma_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
    print()
    print("  >>> MOVING AVERAGE (last {} episodes) <<<".format(window))
    print("  Reward: {:+.2f}  |  Avg length: {:.1f} steps".format(ma_reward, ma_length))
    print("  " + "-" * 50)


def train(
    num_episodes: int = 20,
    seed: Optional[int] = 42,
    agent: str = "random",
    scenario_id: Optional[str] = None,
    llm_model_id: Optional[str] = None,
    llm_method: str = "vanilla",
    best_of_n: int = 5,
    in_context_replay_size: int = 5,
    verbose: bool = False,
    summary_every: int = 10,
    moving_avg_window: int = 10,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Run RL training loop: collect trajectories and print results.

    Args:
        num_episodes: number of episodes to run
        seed: random seed for env (None = random)
        agent: "random", "baseline", or "llm"
        scenario_id: fix scenario (None = random each episode)
        verbose: print per-step details
        summary_every: print periodic summary every N episodes
        moving_avg_window: window size for moving average reward
        plot: show reward plot after training

    Returns:
        Summary dict with episode rewards, lengths, and stats.
        Use result["rewards"] for plotting (e.g. matplotlib).
    """
    env = LifeOpsEnv(seed=seed)

    # Episode logging: rewards, lengths, action sequences
    all_rewards: List[float] = []
    all_lengths: List[int] = []
    all_action_sequences: List[List[str]] = []
    all_scenarios: List[str] = []

    print("=" * 50)
    print("LifeOps RL Training")
    print("=" * 50)
    if agent == "llm" and llm_choose_action is None:
        print("  Warning: LLM agent requested but not available (pip install transformers). Using random.")
        agent = "random"
    print(f"Episodes: {num_episodes}  |  Agent: {agent}  |  Seed: {seed}")
    if agent == "llm" and llm_model_id:
        print(f"  LLM model: {llm_model_id}")
    if agent == "llm" and llm_method != "vanilla":
        print(f"  Method: {llm_method}" + (f" (n={best_of_n})" if llm_method == "best-of-n" else ""))
    print(f"Summary every {summary_every} episodes  |  Moving avg window: {moving_avg_window}")

    # In-context learning: replay buffer of best trajectories
    replay_buffer = None
    in_context_choose_fn = None
    if agent == "llm" and llm_method == "in-context" and InContextReplayBuffer is not None and create_in_context_choose_action is not None:
        # min_reward: only add trajectories better than this (use -inf to always add)
        replay_buffer = InContextReplayBuffer(max_size=in_context_replay_size, min_reward=-999.0)
        in_context_choose_fn = create_in_context_choose_action(
            replay_buffer, num_examples=2, model_id=llm_model_id
        )

    for ep in range(1, num_episodes + 1):
        trajectory, total_reward, ep_len, scenario_id_used, persona_name = collect_trajectory(
            env,
            agent=agent,
            scenario_id=scenario_id,
            llm_model_id=llm_model_id,
            llm_method=llm_method,
            best_of_n=best_of_n,
            in_context_choose_fn=in_context_choose_fn,
        )

        # Update replay buffer for in-context learning
        if replay_buffer is not None and total_reward > 0:
            replay_buffer.add(trajectory, total_reward)

        all_rewards.append(total_reward)
        all_lengths.append(ep_len)
        action_seq = [_format_action_short(t["action"]) for t in trajectory]
        all_action_sequences.append(action_seq)
        all_scenarios.append(scenario_id_used)

        ma_reward = _moving_average(all_rewards, moving_avg_window)

        print_episode_summary(
            episode=ep,
            total_reward=total_reward,
            episode_length=ep_len,
            scenario_id=scenario_id_used,
            persona_name=persona_name,
            trajectory=trajectory,
            moving_avg=ma_reward,
            moving_avg_window=moving_avg_window,
            verbose=verbose,
        )

        if ep % summary_every == 0:
            _print_periodic_summary(ep, all_rewards, all_lengths, moving_avg_window)

    # Final summary
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

    if plot and all_rewards:
        _plot_rewards(all_rewards, window=moving_avg_window, policy=agent)

    return {
        "rewards": all_rewards,
        "lengths": all_lengths,
        "action_sequences": all_action_sequences,
        "scenarios": all_scenarios,
        "avg_reward": avg_reward,
        "avg_length": avg_len,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent on LifeOps")
    parser.add_argument("-n", "--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--agent",
        choices=["random", "baseline", "llm"],
        default="random",
        help="Agent: random, baseline (rule-based), or llm (requires transformers)",
    )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None, help="Fix scenario (e.g. s1_basic_conflict)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model ID (e.g. groq:llama-3.3-70b-versatile). Only for --agent llm.",
    )
    parser.add_argument(
        "--method",
        choices=["vanilla", "best-of-n", "in-context"],
        default="vanilla",
        help="Policy improvement: vanilla (greedy), best-of-n (sample n, pick best), in-context (few-shot from best trajectories)",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=8,
        help="Number of samples for best-of-n method (default: 8)",
    )
    parser.add_argument(
        "--in-context-size",
        type=int,
        default=5,
        help="Replay buffer size for in-context method (default: 5)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--summary-every", type=int, default=10, help="Print summary every N episodes")
    parser.add_argument("--moving-avg-window", type=int, default=10, help="Window size for moving average")
    parser.add_argument("--no-plot", action="store_true", help="Skip reward plot")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        seed=args.seed,
        agent=args.agent,
        scenario_id=args.scenario,
        llm_model_id=args.model,
        llm_method=args.method,
        best_of_n=args.best_of_n,
        in_context_replay_size=args.in_context_size,
        verbose=args.verbose,
        summary_every=args.summary_every,
        moving_avg_window=args.moving_avg_window,
        plot=not args.no_plot,
    )
