#!/usr/bin/env python3
"""
Evaluate and compare agent policies on the LifeOps environment.

Compares: random, baseline, LLM agents.
Metrics: average reward, success rate (conflicts resolved), constraint violations, episode length.
Output: readable table + persona/scenario per episode.
"""

from __future__ import annotations

# Load .env before HuggingFace imports (for HF_TOKEN)
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

from env.lifeops_env import LifeOpsEnv, _choose_simple_action
from env.reward import detect_overlaps, travel_issues
from env.scenario_generator import list_scenario_ids

try:
    from agent.llm_agent import choose_action as llm_choose_action
except ImportError:
    llm_choose_action = None

try:
    from training.train_rl import collect_trajectory
except ImportError:
    collect_trajectory = None


def _run_episode(
    env: LifeOpsEnv,
    agent: str,
    scenario_id: str,
    llm_model_id: Optional[str] = None,
    parse_stats: Optional[Dict[str, int]] = None,
) -> Tuple[float, bool, int, int, str, str]:
    """
    Run one episode for a given agent and scenario.

    Returns:
        total_reward, success (no overlaps + no travel issues at end),
        total_constraint_violations, episode_length, scenario_id, persona_name
    """
    if collect_trajectory is not None:
        trajectory, total_reward, ep_len, sid, persona = collect_trajectory(
            env, agent=agent, scenario_id=scenario_id, llm_model_id=llm_model_id,
            parse_stats=parse_stats,
        )
    else:
        # Fallback: minimal inline run
        obs = env.reset(scenario_id=scenario_id)
        persona = obs["persona"].get("name", "?")
        sid = obs["scenario_id"]
        total_reward = 0.0
        ep_len = 0
        trajectory = []
        done = False

        if agent == "baseline":
            policy_fn = _choose_simple_action
        elif agent == "llm" and llm_choose_action is not None:
            policy_fn = lambda e: llm_choose_action(
                e.observation(), e.valid_actions(),
                model_id=llm_model_id,
                fallback_fn=lambda: _choose_simple_action(e),
                parse_stats=parse_stats,
            )
        else:
            policy_fn = lambda e: random.choice(e.valid_actions())

        while not done:
            action = policy_fn(env)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            ep_len += 1
            trajectory.append({"next_obs": obs, "info": info})

    # Compute success: final calendar has no overlaps and no travel issues
    final_obs = trajectory[-1]["next_obs"] if trajectory else {}
    calendar = final_obs.get("calendar", [])
    travel_times = final_obs.get("travel_times", {})
    home = final_obs.get("persona", {}).get("home_location")
    overlaps = detect_overlaps(calendar)
    issues = travel_issues(calendar, travel_times, start_location=home)
    success = len(overlaps) == 0 and len(issues) == 0

    # Constraint violations: sum over all steps
    total_violations = 0
    for step in trajectory:
        info = step.get("info", {})
        total_violations += len(info.get("overlaps", []))
        total_violations += len(info.get("travel_issues", []))

    return total_reward, success, total_violations, ep_len, sid, persona


def evaluate_agents(
    num_episodes: int = 20,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    agents: Optional[List[str]] = None,
    llm_model_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all agents on the same set of scenarios.

    If seeds is provided, runs each seed separately and aggregates (mean ± std).
    Returns dict: agent_name -> {avg_reward, success_rate, avg_violations, avg_length, episodes}
    """
    if agents is None:
        agents = ["random", "baseline", "llm"]
    if "llm" in agents and llm_choose_action is None:
        agents = [a for a in agents if a != "llm"]
        if agents:
            print("Warning: LLM agent not available (transformers?). Skipping.")
        else:
            agents = ["random", "baseline"]

    seed_list = seeds if seeds is not None else [seed]

    results: Dict[str, Dict[str, Any]] = {}
    for agent in agents:
        rewards: List[float] = []
        successes: List[bool] = []
        violations: List[int] = []
        lengths: List[int] = []
        episode_details: List[Tuple[str, str]] = []  # (persona, scenario)
        llm_parse_stats: Dict[str, int] = {}

        for s in seed_list:
            rng = random.Random(s)
            scenario_ids = [rng.choice(list_scenario_ids()) for _ in range(num_episodes)]
            env = LifeOpsEnv(seed=s + hash(agent) % 1000)
            for ep, scenario_id in enumerate(scenario_ids):
                ep_parse_stats = {} if agent == "llm" else None
                total_reward, success, total_viol, ep_len, sid, persona = _run_episode(
                    env, agent, scenario_id, llm_model_id=llm_model_id,
                    parse_stats=ep_parse_stats,
                )
                rewards.append(total_reward)
                successes.append(success)
                violations.append(total_viol)
                lengths.append(ep_len)
                episode_details.append((persona, sid))
                if ep_parse_stats:
                    for k, v in ep_parse_stats.items():
                        llm_parse_stats[k] = llm_parse_stats.get(k, 0) + v

        extra: Dict[str, Any] = {}
        if agent == "llm" and llm_parse_stats:
            total_decisions = llm_parse_stats.get("parsed", 0) + llm_parse_stats.get("fallback", 0)
            extra["llm_parse_stats"] = llm_parse_stats
            if total_decisions > 0:
                extra["llm_parse_rate_pct"] = 100.0 * llm_parse_stats.get("parsed", 0) / total_decisions

        results[agent] = {
            "avg_reward": sum(rewards) / len(rewards),
            "success_rate": sum(successes) / len(successes) * 100,
            "avg_violations": sum(violations) / len(violations),
            "avg_length": sum(lengths) / len(lengths),
            "rewards": rewards,
            "successes": successes,
            "violations": violations,
            "lengths": lengths,
            "episode_details": episode_details,
            **extra,
        }

    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Print evaluation results in a readable table."""
    print()
    print("=" * 70)
    print("  LifeOps Agent Evaluation")
    print("=" * 70)

    # Summary table
    print("\n  SUMMARY (averaged over episodes)")
    print("  " + "-" * 66)
    header = f"  {'Agent':<12} {'Avg Reward':>12} {'Success %':>10} {'Violations':>12} {'Avg Length':>10}"
    print(header)
    print("  " + "-" * 66)
    for agent, data in results.items():
        row = (
            f"  {agent:<12} "
            f"{data['avg_reward']:>+12.2f} "
            f"{data['success_rate']:>9.1f}% "
            f"{data['avg_violations']:>12.1f} "
            f"{data['avg_length']:>10.1f}"
        )
        print(row)
        if "llm_parse_rate_pct" in data:
            stats = data.get("llm_parse_stats", {})
            print(f"       └─ LLM parse rate: {data['llm_parse_rate_pct']:.1f}% "
                  f"(parsed={stats.get('parsed',0)}, fallback={stats.get('fallback',0)})")
    print("  " + "-" * 66)

    # Episode details (persona + scenario per episode; same for all agents)
    print("\n  EPISODE DETAILS (persona, scenario)")
    print("  " + "-" * 66)
    agents_list = list(results.keys())
    n = len(results[agents_list[0]]["episode_details"])
    for ep in range(n):
        persona, scenario = results[agents_list[0]]["episode_details"][ep]
        print(f"  Ep {ep+1:2d}:  {persona}  |  {scenario}")
    print("=" * 70)


def plot_rewards(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot reward per episode for each agent."""
    try:
        import matplotlib
        if save_path and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"random": "#e74c3c", "baseline": "#27ae60", "llm": "#3498db"}

    for agent, data in results.items():
        rewards = data["rewards"]
        x = list(range(1, len(rewards) + 1))
        color = colors.get(agent, "#95a5a6")
        ax.plot(x, rewards, "o-", label=agent, color=color, markersize=4, linewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward per Episode by Agent")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\n  Plot saved to {save_path}")

    if show:
        plt.show()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LifeOps agents")
    parser.add_argument("-n", "--episodes", type=int, default=20, help="Episodes per agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for robust eval (e.g. --seeds 42 123 456)")
    parser.add_argument("--agents", nargs="+", default=["random", "baseline", "llm"],
                        help="Agents to evaluate")
    parser.add_argument("--plot", action="store_true", default=True, help="Show reward plot")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Skip reward plot")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file")
    parser.add_argument("--model", type=str, default=None, help="LLM model ID (e.g. google/flan-t5-base)")
    args = parser.parse_args()

    seeds_str = f"seeds={args.seeds}" if args.seeds else f"seed={args.seed}"
    print(f"\nEvaluating {args.agents} over {args.episodes} episodes ({seeds_str})...")
    if args.model:
        print(f"  LLM model: {args.model}")
    results = evaluate_agents(
        num_episodes=args.episodes,
        seed=args.seed,
        seeds=args.seeds,
        agents=args.agents,
        llm_model_id=args.model,
    )
    print_results(results)

    if args.plot or args.save_plot:
        plot_rewards(results, save_path=args.save_plot, show=args.plot)


if __name__ == "__main__":
    main()
