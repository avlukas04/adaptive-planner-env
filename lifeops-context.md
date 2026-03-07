# LifeOps Project Context

You are helping build an MVP for an OpenEnv hackathon project called **LifeOps**.

LifeOps is a reinforcement learning environment where an AI agent learns to manage a user's life schedule.

The environment simulates calendar management, personal goals, travel constraints, and incoming requests.

The purpose is to train agents to make realistic planning decisions using reinforcement learning.

---

# Core Project Goal

Build a **personalized planning environment** where an AI agent manages:

* calendar events
* personal goals/tasks
* travel constraints
* incoming messages or scheduling requests

The environment should expose:

* a structured state
* a constrained action space
* a reward function

This allows reinforcement learning algorithms to train agents to improve planning decisions.

---

# MVP Scope

Keep the system **simple and hackathon-friendly**.

Constraints:

* Python only
* No unnecessary frameworks
* Clean modular structure
* Readable code

Design decisions:

* Use **structured JSON-like state** instead of raw text
* Use a **one-day planning horizon**
* Include **user personas** with preferences and behavioral tendencies

Environment should include:

* calendar events
* tasks/goals
* travel times
* incoming requests/messages

---

# Action Space

The agent should choose structured actions such as:

* accept_event
* reject_event
* reschedule_event
* propose_new_time
* block_focus_time

Avoid free-form natural language actions.

---

# Reward Design

Penalize:

* double booking
* impossible travel
* violating strong user preferences
* missing important obligations

Reward:

* resolving scheduling conflicts
* respecting user habits
* allocating time to goals
* correctly handling important requests

---

# Coding Rules

Follow these rules when generating code:

* Keep files small and modular
* Add docstrings and comments
* Prefer dataclasses or typed dictionaries for structured data
* Avoid unnecessary abstractions
* Avoid complex frameworks
* Prefer clarity over cleverness

Add basic input validation and avoid unsafe patterns.

Follow general OWASP best practices where applicable.

---

# Initial Repository Targets

env/personas.py
env/actions.py
env/scenario_generator.py
env/reward.py
env/lifeops_env.py

tests/test_env.py
tests/test_reward.py

---

# Validation Requirements

The code must:

* run locally
* include a simple manual episode runner
* include at least **3 personas**
* include at least **5 sample scenarios**
* include at least **5 tests**

Tests should verify:

* reward calculation
* environment step logic
* conflict detection
* travel feasibility
* persona preference handling

---

# Implementation Philosophy

This project is a **simulation environment**, not a full product.

Focus on:

* clarity
* correctness
* simple reinforcement learning compatibility

Do not build UI unless explicitly requested.

If something is ambiguous, make a reasonable assumption and document it.
