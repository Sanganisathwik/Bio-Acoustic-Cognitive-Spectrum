import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from agents import fixed_agent, random_agent, rule_based_agent
from audio_utils import audio_to_spectrogram
from env import N_CHANNELS, generate_state, step
from train_cnn import cnn_history, train
from train_dqn import dqn_history, train_dqn

# Set visual style
sns.set_theme(style="whitegrid")


def plot_reward_heatmap():
    noise_range = np.linspace(0, 1, 20)
    traffic_range = np.linspace(0, 1, 20)
    # Assuming bio = 0 for the base map
    rewards = [
        [3 * (1 - 0) - n * 1.0 - t * 0.5 for t in traffic_range] for n in noise_range
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        rewards,
        xticklabels=np.round(traffic_range, 1),
        yticklabels=np.round(noise_range, 1),
    )
    plt.title("Reward Map (Bio=0)")
    plt.xlabel("Traffic")
    plt.ylabel("Noise")
    plt.savefig("output/reward_heatmap.png")


def plot_channel_state():
    state = generate_state()
    df = pd.DataFrame(state, columns=["Noise", "Bio", "Traffic"])
    df["Channel"] = [f"CH {i}" for i in range(N_CHANNELS)]

    df.set_index("Channel").plot(kind="bar", figsize=(10, 5))
    plt.title("Current Environment State Snapshot")
    plt.ylabel("Magnitude (0-1)")
    plt.savefig("output/channel_state.png")


def plot_spectrogram(file_path):
    spec_db = audio_to_spectrogram(file_path)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec_db, x_axis="time", y_axis="mel", sr=16000)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-Spectrogram: {file_path}")
    plt.savefig("output/spectrogram.png")


def plot_cnn_curves(history):
    # history should be a dict with 'loss' and 'acc' lists from train_cnn.py
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"], label="Loss")
    ax1.set_title("CNN Training Loss")
    ax2.plot(history["acc"], label="Accuracy", color="orange")
    ax2.set_title("CNN Training Accuracy")
    plt.savefig("output/cnn_training.png")


def plot_agent_comparison():
    agents = {
        "Random": random_agent,
        "Fixed": fixed_agent,
        "Rule-Based": rule_based_agent,
    }

    results = []
    for name, agent in agents.items():
        total_reward = 0
        total_interference = 0
        trials = 100
        for _ in range(trials):
            state = generate_state()
            action = agent(state)
            total_reward += step(state, action)
            if state[action][1] == 1:  # Bio presence check
                total_interference += 1

        results.append(
            {
                "Agent": name,
                "Avg Reward": total_reward / trials,
                "Interference %": (total_interference / trials) * 100,
            }
        )

    df = pd.DataFrame(results)

    # Plotting Avg Reward
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Agent", y="Avg Reward", data=df, palette="viridis")
    plt.title("Average Reward by Agent Type")
    plt.savefig("output/agent_rewards.png")

    # Plotting Interference Rate
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Agent", y="Interference %", data=df, palette="magma")
    plt.title("Biological Interference Rate (%)")
    plt.savefig("output/interference_rate.png")


def plot_dqn_training(rewards_history, actions_history):
    # Reward vs Episode
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    # Rolling average to see trend
    rolling_avg = pd.Series(rewards_history).rolling(window=20).mean()
    plt.plot(rolling_avg, color="red", linewidth=2)
    plt.title("DQN Learning Curve (Reward vs Episode)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("output/dqn_learning.png")

    # Action Distribution (Pie Chart)
    plt.figure(figsize=(6, 6))
    unique, counts = np.unique(actions_history[-500:], return_counts=True)
    plt.pie(counts, labels=[f"CH {i}" for i in unique], autopct="%1.1f%%")
    plt.title("Action Distribution (Last 500 Steps)")
    plt.savefig("output/action_distribution.png")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    plot_channel_state()
    plot_reward_heatmap()
    
    # Pick a valid bio file dynamically
    bio_files = [f for f in os.listdir("data/bio") if f.endswith(".wav")]
    if bio_files:
        plot_spectrogram(os.path.join("data/bio", bio_files[0]))
    plot_agent_comparison()
    train()
    plot_cnn_curves(cnn_history)
    plot_agent_comparison()
    train_dqn(500)
    plot_dqn_training(dqn_history["rewards"], dqn_history["actions"])
