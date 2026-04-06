import random
from collections import deque

import torch
import torch.nn as nn
from dqn import create_model, flatten_state
from env import generate_state, step

N_CHANNELS = 5

dqn_history = {
    "rewards": [],
    "actions": [],
}


def train_dqn(episodes=200):
    state_size = N_CHANNELS * 3
    action_size = N_CHANNELS

    model = create_model(state_size, action_size)
    target_model = create_model(state_size, action_size)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=2000)

    gamma = 0.95
    epsilon = 1.0

    for episode in range(episodes):
        state = generate_state()
        total_reward = 0

        for step_num in range(50):
            state_vec = torch.FloatTensor(flatten_state(state))

            # ε-greedy
            if random.random() < epsilon:
                action = random.randint(0, N_CHANNELS - 1)
            else:
                q_values = model(state_vec)
                action = torch.argmax(q_values).item()

            reward = step(state, action)
            next_state = generate_state()

            replay_buffer.append((state, action, reward, next_state))

            state = next_state
            total_reward += reward

            # train
            if len(replay_buffer) > 32:
                batch = random.sample(replay_buffer, 32)

                states = torch.FloatTensor([flatten_state(b[0]) for b in batch])
                actions = torch.LongTensor([b[1] for b in batch])
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([flatten_state(b[3]) for b in batch])

                q_values = model(states)
                next_q_values = target_model(next_states)

                target_q = q_values.clone()

                for i in range(32):
                    target = rewards[i] + gamma * torch.max(next_q_values[i])
                    target_q[i][actions[i]] = target

                loss = loss_fn(q_values, target_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Logging for visualization
            dqn_history["rewards"].append(total_reward)
            dqn_history["actions"].append(action)

        epsilon = max(0.05, epsilon * 0.995)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode}, Reward: {total_reward:.2f}")

    torch.save(model.state_dict(), "data/dqn_model.pth")

    return model


if __name__ == "__main__":
    train_dqn(episodes=500)
