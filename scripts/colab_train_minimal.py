"""
Minimal LifeOps training script for Colab using HuggingFace TRL.

Run in Colab:
  !pip install -q transformers trl torch
  !git clone https://github.com/YOUR_USER/adaptive-planner-env.git
  %run adaptive-planner-env/scripts/colab_train_minimal.py

Or paste this into a Colab cell.
"""

# %% [markdown]
# # LifeOps + HF TRL — Minimal Training
# Trains an LLM policy on the LifeOps scheduling environment using reward-based optimization.

# %%
!pip install -q transformers trl torch python-dotenv

# %%
import os
import sys
from pathlib import Path

# Clone or mount repo (adjust path for your setup)
REPO = Path("adaptive-planner-env")
if not REPO.exists():
    !git clone https://github.com/YOUR_USER/adaptive-planner-env.git 2>/dev/null || true
sys.path.insert(0, str(REPO))

# %%
from env.lifeops_env import LifeOpsEnv
from env.scenario_generator import list_scenario_ids
from training.train_rl import collect_trajectory, train

# TRL: we use the env's reward signal for policy improvement
# This script demonstrates TRL integration — full PPO/DPO would require log-probs from the model
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler

# %%
# Collect trajectories from LifeOps env (reward signal for TRL-style optimization)
env = LifeOpsEnv(seed=42)
trajectories = []
for _ in range(5):
    traj, reward, length, sid, persona = collect_trajectory(env, agent="baseline")
    trajectories.append({"trajectory": traj, "reward": reward, "scenario_id": sid})

print(f"Collected {len(trajectories)} trajectories")
print(f"Sample reward: {trajectories[0]['reward']:.2f}")

# %%
# Train with LLM agent (uses reward from env; TRL provides the optimization framework)
result = train(
    num_episodes=10,
    agent="llm",
    llm_model_id="google/flan-t5-base",  # Small model for Colab
    llm_method="vanilla",
    plot=False,
)
print(f"Avg reward: {result['avg_reward']:.2f}")
