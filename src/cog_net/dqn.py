import torch
import torch.nn as nn


def create_model(state_size, action_size):
    model = nn.Sequential(
        nn.Linear(state_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_size),
    )
    return model


def flatten_state(state):
    flat = []
    for noise, bio, traffic in state:
        flat.extend([noise, bio, traffic])
    return flat
