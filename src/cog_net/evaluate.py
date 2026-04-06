from agents import fixed_agent, random_agent, rule_based_agent
from env import generate_state, step


# Evaluation function
def evaluate(agent_fn, episodes=100, steps=50):
    total_reward = 0
    total_interference = 0
    total_steps = 0

    for _ in range(episodes):
        state = generate_state()

        for _ in range(steps):
            action = agent_fn(state)
            reward = step(state, action)

            # unpack state info
            _, bio, _ = state[action]

            if bio == 1:
                total_interference += 1

            total_reward += reward
            total_steps += 1

            state = generate_state()

    avg_reward = total_reward / total_steps
    interference_rate = total_interference / total_steps

    return avg_reward, interference_rate


if __name__ == "__main__":
    agents = {
        "Random": random_agent,
        "Fixed": fixed_agent,
        "Rule-Based": rule_based_agent,
    }

    for name, agent in agents.items():
        avg_reward, interference = evaluate(agent)

        print(f"{name} Agent:")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Interference Rate: {interference:.2f}")
        print()
