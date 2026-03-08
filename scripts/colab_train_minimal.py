"""
Minimal LifeOps training script for Colab (Unsloth or HF TRL).

Copy each # Cell block into a Colab cell and run. Uses the LifeOps RL environment
with HF TRL SFTTrainer for policy improvement via supervised fine-tuning on
high-reward trajectories.

For faster training, optionally use Unsloth: pip install unsloth
"""

# =============================================================================
# Cell 1: Install dependencies
# =============================================================================
# !pip install -q transformers trl torch datasets accelerate python-dotenv
# # Optional: Unsloth for 2x faster LoRA training
# # !pip install -q unsloth

# =============================================================================
# Cell 2: Clone repo and setup path
# =============================================================================
# !git clone -q https://github.com/avlukas04/adaptive-planner-env.git
# import sys
# sys.path.insert(0, "adaptive-planner-env")

# =============================================================================
# Cell 3: Train with HF TRL SFTTrainer (LifeOps env + reward signal)
# =============================================================================
"""
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from env.lifeops_env import LifeOpsEnv
from training.train_rl import collect_trajectory
from agent.llm_agent import _state_to_prompt
from env.actions import generate_valid_actions

# 1. Collect high-reward trajectories from LifeOps env
env = LifeOpsEnv(seed=42)
rows = []
for _ in range(20):
    traj, reward, _, _, _ = collect_trajectory(env, agent="baseline")
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

# 2. SFT with TRL (flan-t5-small for Colab; use flan-t5-base or Phi-3 for better results)
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

config = SFTConfig(
    output_dir="./lifeops_sft",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    max_seq_length=256,
)
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./lifeops_sft")
print("Done. Model saved to ./lifeops_sft")
"""

# =============================================================================
# Cell 4 (alternative): Run train_rl loop only (no TRL, env + Best-of-N / in-context)
# =============================================================================
"""
# Simplest option: run the built-in RL training loop
from training.train_rl import train

result = train(num_episodes=10, agent="baseline", plot=False)
print(f"Avg reward: {result['avg_reward']:.2f}")
"""
