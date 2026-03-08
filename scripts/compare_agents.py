import sys
sys.path.insert(0, '.')
from env.lifeops_env import LifeOpsEnv, _choose_simple_action
from agent.llm_agent import choose_action, FEW_SHOT_PREFIX

SCENARIOS = ['s1_basic_conflict', 's2_travel_tight', 's3_focus_vs_meeting', 's4_parent_pickup', 's5_late_meeting']
MODEL = 'groq:llama-3.1-8b-instant'

def run_episode(scenario_id, use_llm=False):
    env = LifeOpsEnv()
    env.reset(scenario_id)
    done, total = False, 0
    parse_stats = {'parsed': 0, 'fallback': 0}
    while not done:
        state = env.observation()
        actions = env.valid_actions()
        if use_llm:
            action = choose_action(state, actions, model_id=MODEL,
                                   few_shot_prefix=FEW_SHOT_PREFIX,
                                   parse_stats=parse_stats)
        else:
            action = _choose_simple_action(env)
        _, reward, done, _ = env.step(action)
        total += reward
    return total, parse_stats

print(f"{'Scenario':<25} {'Baseline':>10} {'LLM':>10} {'Delta':>10} {'ParseRate':>12}")
print("-" * 70)
for s in SCENARIOS:
    baseline_r, _ = run_episode(s, use_llm=False)
    llm_r, stats = run_episode(s, use_llm=True)
    total = stats['parsed'] + stats['fallback']
    rate = stats['parsed'] / total if total > 0 else 0
    delta = llm_r - baseline_r
    print(f"{s:<25} {baseline_r:>10.2f} {llm_r:>10.2f} {delta:>+10.2f} {rate:>11.0%}")
