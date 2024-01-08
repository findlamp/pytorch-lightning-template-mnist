import torch
import numpy as np
from torch import nn
from . import common

class SimpleNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=10, hid=128, layer_num=5):
        super().__init__()
        
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    def forward(self, x):
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
