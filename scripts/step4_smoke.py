from env.lifeops_env import LifeOpsEnv
env = LifeOpsEnv()
for sid in ['s1_basic_conflict', 's2_travel_tight', 's3_focus_vs_meeting']:
    obs = env.reset(sid)
    done, total, steps = False, 0, 0
    while not done:
        actions = env.valid_actions()
        obs, reward, done, info = env.step(actions[0])
        total += reward
        steps += 1
    print(f'{sid}: steps={steps} reward={total:.2f}')
print('OK')
