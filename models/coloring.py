import torch.nn as nn
from torch.nn import Sequential


class Coloring(nn.Module):
    def __init__(self, n_colors):
        super().__init__()
        self.color = Sequential(
            nn.LazyLinear(n_colors),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.color(x)
