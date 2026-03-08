"""
Minimal LifeOps training script for Colab (OpenEnv 0.2.1 + Unsloth/TRL).

Uses OpenEnv (stable 0.2.1) deployed on HF Spaces. Copy each # Cell block
into a Colab cell and run.

Requirements:
- OpenEnv env deployed at: https://YOUR_ORG-lifeops-env.hf.space
  (Create a Docker Space, push Dockerfile.openenv, env/, openenv_lifeops/)
"""

# =============================================================================
# Cell 1: Install dependencies (OpenEnv 0.2.1 + TRL)
# =============================================================================
# !pip install -q openenv-core==0.2.1 transformers trl torch datasets accelerate python-dotenv
# # Optional: Unsloth for 2x faster LoRA
# # !pip install -q unsloth

# =============================================================================
# Cell 2: Clone repo and setup path
# =============================================================================
# !git clone -q https://github.com/avlukas04/adaptive-planner-env.git
# import sys
# sys.path.insert(0, "adaptive-planner-env")

# =============================================================================
# Cell 3a: Use OpenEnv remote env (connects to HF Space)
# =============================================================================
"""
# Set your LifeOps OpenEnv Space URL (after deploying)
LIFEOPS_ENV_URL = "https://openenv-community-lifeops-env.hf.space"

from openenv_lifeops.env_adapter import LifeOpsEnvAdapter
from env.lifeops_env import _choose_simple_action
from training.train_rl import collect_trajectory

# Connect to OpenEnv env on HF Spaces
env = LifeOpsEnvAdapter(base_url=LIFEOPS_ENV_URL)

# Collect trajectories (uses remote env)
trajectories = []
for ep in range(10):
    traj, reward, _, _ = collect_trajectory(env, policy="heuristic")
    print(f"Episode {ep+1}: reward={reward:.2f}")
    if reward > 0:
        trajectories.append((traj, reward))

env.close()
print(f"Collected {len(trajectories)} positive-reward trajectories")
"""

# =============================================================================
# Cell 3b: Use local env (no deployment; for quick testing)
# =============================================================================
"""
from env.lifeops_env import LifeOpsEnv, _choose_simple_action
from training.train_rl import collect_trajectory

env = LifeOpsEnv(seed=42)
trajectories = []
for ep in range(10):
    traj, reward, _, _ = collect_trajectory(env, policy="heuristic")
    print(f"Episode {ep+1}: reward={reward:.2f}")
    if reward > 0:
        trajectories.append((traj, reward))
print(f"Collected {len(trajectories)} positive-reward trajectories")
"""

# =============================================================================
# Cell 4: Train with HF TRL SFTTrainer (OpenEnv env + reward signal)
# =============================================================================
"""
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from env.lifeops_env import LifeOpsEnv
from env.actions import generate_valid_actions
from agent.llm_agent import _state_to_prompt
from training.train_rl import collect_trajectory

# Collect trajectories (local or use LifeOpsEnvAdapter for remote)
env = LifeOpsEnv(seed=42)
rows = []
for _ in range(20):
    traj, reward, _, _ = collect_trajectory(env, policy="heuristic")
    if reward <= 0:
        continue
    for step in traj[:4]:
        obs = step["obs"]
        action_dict = step["action"]
        valid = generate_valid_actions(obs)
        prompt = _state_to_prompt(obs, valid)
        at = action_dict.get("action_type", "?")
        idx = next((i + 1 for i, a in enumerate(valid) if a.to_dict().get("action_type") == at), 1)
        completion = f"Reasoning: Aligned with persona preferences. CHOICE: {idx}"
        rows.append({"text": prompt + "\n" + completion})

if not rows:
    rows = [{"text": "Persona: Early-bird\nCalendar: empty\nOptions: 1. accept 2. reject\nCHOICE: 1"}]

dataset = Dataset.from_list(rows)
print(f"Dataset size: {len(dataset)}")

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

config = SFTConfig(output_dir="./lifeops_sft", num_train_epochs=1, per_device_train_batch_size=2, max_seq_length=256)
trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, dataset_text_field="text", tokenizer=tokenizer)
trainer.train()
trainer.save_model("./lifeops_sft")
print("Done. Model saved to ./lifeops_sft")
"""

# =============================================================================
# Cell 5: Run train_rl loop (local or OpenEnv)
# =============================================================================
"""
from training.train_rl import train

result = train(num_episodes=10, policy="heuristic")
print(f"Avg reward: {result['avg_reward']:.2f}")
"""
