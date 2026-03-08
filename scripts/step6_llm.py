import sys
sys.path.insert(0, '.')
from env.lifeops_env import LifeOpsEnv
from agent.llm_agent import choose_action, FEW_SHOT_PREFIX

parse_stats = {'parsed': 0, 'fallback': 0}
env = LifeOpsEnv()
env.reset('s1_basic_conflict')
done, total = False, 0

while not done:
    state = env.observation()
    actions = env.valid_actions()
    action = choose_action(
        state, actions,
        model_id='groq:llama-3.1-8b-instant',
        few_shot_prefix=FEW_SHOT_PREFIX,
        parse_stats=parse_stats,
    )
    print(f'  -> {action.action_type.value}')
    obs, reward, done, info = env.step(action)
    total += reward
    print(f'     reward={reward:.2f} breakdown={info.get("reward_breakdown", {})}')

print(f'Total={total:.2f}')
print(f'ParseStats={parse_stats}')
rate = parse_stats["parsed"] / max(1, parse_stats["parsed"] + parse_stats["fallback"])
print(f'ParseRate={rate:.0%}')
