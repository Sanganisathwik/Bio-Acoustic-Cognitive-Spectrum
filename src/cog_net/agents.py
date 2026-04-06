import random

from env import N_CHANNELS, generate_state, step


# Random agent
def random_agent(state):
    return random.randint(0, N_CHANNELS - 1)


# Fixed agent (always chooses channel 0)
def fixed_agent(state):
    return 0


# Rule-based agent (IMPORTANT)
def rule_based_agent(state):
    best_score = float("inf")
    best_action = 0

    for i, (noise, bio, traffic) in enumerate(state):
        # lower score = better channel
        score = noise + traffic + (bio * 10)

        if score < best_score:
            best_score = score
            best_action = i

    return best_action


def create_model(state_size, action_size):
    model = nn.Sequential(
        nn.Linear(state_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_size),
    )
    return model


if __name__ == "__main__":
    state = generate_state()

    print("State:")
    for i, (noise, bio, traffic) in enumerate(state):
        print(f"Channel {i}: noise={noise:.2f}, bio={bio}, traffic={traffic:.2f}")

    agents = {
        "Random": random_agent,
        "Fixed": fixed_agent,
        "Rule-Based": rule_based_agent,
    }

    print("\nAgent Decisions:")
    for name, agent in agents.items():
        action = agent(state)
        reward = step(state, action)

        print(f"{name} Agent → Action: {action}, Reward: {reward:.2f}")
