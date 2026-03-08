from agent.llm_agent import parse_llm_action
from env.actions import Action, ActionType

actions = [
    Action(action_type=ActionType.accept_event, request_id="r1"),
    Action(action_type=ActionType.reject_event, request_id="r1"),
    Action(action_type=ActionType.block_focus_time, new_start_min=540, duration_min=60),
]

test_cases = [
    ("CHOICE: 1", 0),
    ("choice: 2", 1),
    ("I think option CHOICE: 3 is best", 2),
    ("2", 1),
    ("The answer is 1", 0),
    ("accept_event", 0),
    ("gibberish xyz", None),
    ("", None),
]

passed = 0
for text, expected_idx in test_cases:
    result = parse_llm_action(text, actions)
    expected = actions[expected_idx] if expected_idx is not None else None
    status = "✓" if result == expected else "✗"
    got_idx = actions.index(result) if result in actions else None
    print(f"{status} input={repr(text):<40} expected={expected_idx}  got={got_idx}")
    if result == expected:
        passed += 1

print(f"\n{pd}/{len(test_cases)} passed")
