import numpy as np

N_CHANNELS = 5


def generate_state():
    state = []

    for _ in range(N_CHANNELS):
        noise = np.random.rand()  # 0 to 1
        bio = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% chance bio
        traffic = np.random.rand()  # 0 to 1

        state.append((noise, bio, traffic))

    return state


def step(state, action):
    noise, bio, traffic = state[action]

    reward = 0

    reward = 3 * (1 - bio)  # reward avoiding bio
    reward -= noise * 1.0
    reward -= traffic * 0.5

    return reward


if __name__ == "__main__":
    state = generate_state()

    print("Generated State:")
    for i, (noise, bio, traffic) in enumerate(state):
        print(f"Channel {i}: noise={noise:.2f}, bio={bio}, traffic={traffic:.2f}")

    action = 4  # pick any channel manually
    reward = step(state, action)

    print(f"\nAction taken: {action}")
    print(f"Reward: {reward:.2f}")
