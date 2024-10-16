import torch
import torch.nn as nn


class Projection(torch.nn.Module):
    def __init__(self):
        super(Projection, self).__init__()

        self.backdone = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
        )

    def forward(self, x):
        y = self.backdone(x)
        return y
