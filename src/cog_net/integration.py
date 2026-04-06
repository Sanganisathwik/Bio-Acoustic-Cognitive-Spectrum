import os
import random

import numpy as np
import torch
from audio_utils import audio_to_spectrogram, fix_size
from cnn import create_cnn
from dqn import create_model, flatten_state
from env import N_CHANNELS, step


def get_bio_from_audio(model, file):
    spec = audio_to_spectrogram(file)
    spec = fix_size(spec)

    spec = np.nan_to_num(spec)

    min_val = spec.min()
    max_val = spec.max()

    if max_val - min_val > 0:
        spec = (spec - min_val) / (max_val - min_val)
    else:
        spec = np.zeros_like(spec)

    spec = np.expand_dims(spec, axis=0)

    x = torch.FloatTensor(spec).unsqueeze(0)

    with torch.no_grad():
        pred = model(x)

    return pred.item()


def generate_state_with_cnn(model, bio_files, nonbio_files):
    state = []

    for _ in range(N_CHANNELS):
        noise = np.random.rand()
        traffic = np.random.rand()

        if random.random() < 0.5:
            file = random.choice(bio_files)
        else:
            file = random.choice(nonbio_files)

        bio_prob = get_bio_from_audio(model, file)

        state.append((noise, bio_prob, traffic))

    return state


def run_integration(num_episodes=10):
    # load CNN
    cnn_model = create_cnn()
    cnn_model.load_state_dict(torch.load("data/cnn_model.pth"))
    cnn_model.eval()

    # load audio file paths
    bio_files = [os.path.join("data/bio", f) for f in os.listdir("data/bio")]
    nonbio_files = [os.path.join("data/nonbio", f) for f in os.listdir("data/nonbio")]

    # load trained DQN (if saved)
    dqn_model = create_model(15, 5)

    # run simulation
    for episode in range(num_episodes):
        state = generate_state_with_cnn(cnn_model, bio_files, nonbio_files)

        state_vec = torch.FloatTensor(flatten_state(state))
        q_values = dqn_model(state_vec)
        q_np = q_values.detach().numpy()
        print("Q:", q_np)

        if random.random() < 0.1:
            action = random.randint(0, 4)
        else:
            action = torch.argmax(q_values).item()

        reward = step(state, action)

        print(f"Episode {episode}: Action={action}, Reward={reward:.2f}")
