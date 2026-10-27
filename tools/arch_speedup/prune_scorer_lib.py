import torch
from torch import nn

class Probe(nn.Module):
    def __init__(self, d=768, hidden=0):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.net = nn.Linear(d, 1) if not hidden else nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(self.norm(x.float())).squeeze(-1)


