import torch.nn as nn


def create_cnn():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Dropout(0.1), # Minimal dropout for maximum accuracy
        nn.Linear(32 * 32 * 32, 256), # Increased capacity from 64 to 256
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model
