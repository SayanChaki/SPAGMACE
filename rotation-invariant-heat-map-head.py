import torch
from torch import nn
from torch.nn import functional as F
from rotationinvariantheatmaphead import *
class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.kernel_size % 2 == 0:
            pad_h = (self.kernel_size // 2, self.kernel_size // 2 - 1)
            pad_w = (self.kernel_size // 2, self.kernel_size // 2 - 1)
        else:
            pad_h = (self.kernel_size // 2, self.kernel_size // 2)
            pad_w = (self.kernel_size // 2, self.kernel_size // 2)
        
        x_padded = F.pad(x, (*pad_w, *pad_h), mode='circular')
        return self.conv(x_padded)

class RotationInvariantNetwork(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, topology):
        super().__init__()
        self.topology = topology
        self.net = self._build_backbone(n_in_channels, n_out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def _build_backbone(self, n_in_channels, n_out_channels):
        layers = []
        n_prev = n_in_channels
        for layer in self.topology:
            layers.append(CircularConv2d(n_prev, layer['filters'], layer['kernel_size']))
            layers.append(nn.ReLU())
            n_prev = layer['filters']
        layers.append(CircularConv2d(n_prev, n_out_channels, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = self.global_pool(x)
        return x.squeeze(-1).squeeze(-1)

def get_heat_map_head():
    n_features = cfg.n_backbone_features
    n_class = 1
    
    topology = [
        dict(filters=128, kernel_size=3),
        dict(filters=128, kernel_size=3),
        dict(filters=128, kernel_size=3),
    ]
     
    return RotationInvariantNetwork(n_features, n_class, topology)
