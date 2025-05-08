from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear, Module, ModuleDict, ModuleList
from torch.nn import functional as F

from gmair.config import config as cfg
from gmair.utils import debug_tools


        
class BasicNetwork(Module):
    def __init__(self, n_in_channels, n_out_channels, topology, internal_activation=ReLU):
        '''
        Builds CNN
        :param n_in_channels:
        :param n_out_channels:
        :param topology:
        :param internal_activation:
        :return:
        '''
        super().__init__()
        self.topology = topology
        
        self.net = self._build_backbone(n_in_channels, n_out_channels)

    def _build_backbone(self, n_in_channels, n_out_channels):
        '''Builds the convnet of the backbone'''

        n_prev = n_in_channels
        net = OrderedDict()

        # Builds internal layers except for the last layer
        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev

            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f #rename
            else:
                f = layer['out_channels']

            net['conv_%d' % i] = Conv2d(**layer)
            net['act_%d' % i] = ReLU()
            n_prev = f

        # Builds the final layer
        net['conv_out'] = Conv2d(in_channels=f, out_channels=n_out_channels, kernel_size=1, stride=1)
                
        return Sequential(net)

    def forward(self, x):
        x = self.net(x)
        return x
        
'''def get_heat_map_head():
    n_features = cfg.n_backbone_features
    
    n_class = 1
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_class, topology)'''
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotationEquivariantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_rotations=8, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations
        self.stride = stride
        self.padding = padding

        # Create a learnable base convolutional filter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        device = x.device  # Ensure we are working on the same device as the input tensor

        # Get filter rotations
        rotated_weights = self.get_rotated_weights(self.weight.to(device))

        # Apply convolution for each rotated filter and stack the results
        out = []
        for i in range(self.num_rotations):
            conv = F.conv2d(x, rotated_weights[i], stride=self.stride, padding=self.padding)
            out.append(conv)

        # Stack along a new dimension representing rotations
        out = torch.stack(out, dim=1)

        # Max-pooling over the rotation dimension to introduce rotation invariance
        out, _ = torch.max(out, dim=1)
        
        return out

    def rotate_filter(self, filter, angle, device):
        # Create the rotation matrix and ensure it's on the same device as the filter
        theta = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float, device=device)  # Move to the correct device

        # Create the affine grid for rotation and move it to the same device
        grid = F.affine_grid(theta.unsqueeze(0), filter.unsqueeze(0).size(), align_corners=False)

        # Rotate the filter using grid_sample
        rotated_filter = F.grid_sample(filter.unsqueeze(0), grid, align_corners=False)

        return rotated_filter.squeeze(0)

    def get_rotated_weights(self, weights):
        device = weights.device  # Ensure that the weights are on the correct device
        rotated_weights = []

        # Rotate the convolution filters by different angles
        for i in range(self.num_rotations):
            angle = (2 * math.pi * i) / self.num_rotations
            rotated_filters = []
            for j in range(weights.size(0)):  # For each output channel
                rotated_filter = self.rotate_filter(weights[j], angle, device)
                rotated_filters.append(rotated_filter)
            rotated_weights.append(torch.stack(rotated_filters))

        return rotated_weights


import torch
import torch.nn as nn

class RotationInvariantGridNetwork(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, topology, num_rotations=8):
        super().__init__()
        self.num_rotations = num_rotations
        self.topology = topology
        self.net = self._build_backbone(n_in_channels, n_out_channels)

    def _build_backbone(self, n_in_channels, n_out_channels):
        n_prev = n_in_channels
        net = nn.ModuleDict()

        # Building the rotation-equivariant layers according to the given topology
        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev
            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f
            else:
                f = layer['out_channels']

            # Apply rotation-equivariant convolution
            net[f'rot_conv_{i}'] = RotationEquivariantConv2d(**layer, num_rotations=self.num_rotations)
            net[f'act_{i}'] = nn.ReLU()

            # No pooling layers, so we just update the number of channels
            n_prev = f

        # Final output convolution to produce logits with 1 channel
        net['conv_out'] = RotationEquivariantConv2d(
            in_channels=f,
            out_channels=n_out_channels,
            kernel_size=1,
            stride=1,
            num_rotations=self.num_rotations
        )

        return net

    def forward(self, x):
        for name, layer in self.net.items():
            x = layer(x)
            x = torch.sigmoid(x)
        return x


def get_heat_map_head():
    device = 'cuda'
    n_features = cfg.n_backbone_features  # Number of input features (from the backbone)
    n_class = 1  # Since we are predicting binary class (object/no object in grid)
    
    # Topology defining the convolutional layers
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (16x16)
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (16x16)
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (16x16)
    ]
    
    # Create the rotation-invariant grid network
    model = RotationInvariantGridNetwork(n_features, n_class, topology)
    
    return model.to(device)



import torch
import torch.nn as nn
from collections import OrderedDict

class ModifiedWhereNetwork(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, topology):
        super().__init__()
        self.topology = topology
        self.net = self._build_backbone(n_in_channels, n_out_channels)

    def _build_backbone(self, n_in_channels, n_out_channels):
        n_prev = n_in_channels
        net = OrderedDict()
        
        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev
            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f
            else:
                f = layer['out_channels']
            
            net[f'conv_{i}'] = nn.Conv2d(**layer)
            net[f'bn_{i}'] = nn.BatchNorm2d(f)
            net[f'act_{i}'] = nn.LeakyReLU(0.2)
            net[f'dropout_{i}'] = nn.Dropout2d(0.1)
            
            n_prev = f

        # Modified output layer
        net['conv_out'] = nn.Conv2d(in_channels=f, out_channels=n_out_channels, kernel_size=1, stride=1)
        
        return nn.Sequential(net)

    def forward(self, x):
        features = self.net(x)
        
        # Separate the output into different components
        mean_xy = features[:, :2]
        log_var_xy = features[:, 2:4]
        mean_hw = features[:, 4:6]
        log_var_hw = features[:, 6:8]
        mean_theta = features[:, 8:9]
        log_var_theta = features[:, 9:]
        
        # Apply appropriate activations/constraints
        mean_xy = torch.sigmoid(mean_xy)  # Constrain to [0, 1]
        mean_hw = torch.exp(mean_hw)  # Ensure positive values
        mean_theta = torch.tanh(mean_theta) * torch.pi  # Constrain to [-pi, pi]
        
        # Ensure variance is positive and not too small
        log_var_xy = torch.clamp(log_var_xy, min=-10, max=10)
        log_var_hw = torch.clamp(log_var_hw, min=-10, max=10)
        log_var_theta = torch.clamp(log_var_theta, min=-10, max=10)
        
        return torch.cat([mean_xy, log_var_xy, mean_hw, log_var_hw, mean_theta, log_var_theta], dim=1)

def get_where_head():
    n_features = cfg.n_backbone_features
    n_localization_latent = 10  # 4 for (y, x, h, w) + 2 for theta (mean, var)
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
    ]
    return ModifiedWhereNetwork(n_features, n_localization_latent, topology)
        
def get_depth_head():
    n_features = cfg.n_backbone_features

    n_depth_latent = 2
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_depth_latent, topology)

# [B, chan * obj_size * obj_size + num_classes + 1] -> [B, 2 * N_ATTR] 
class ObjectEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(ObjectEncoder, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        ) 
        
    
    def forward(self, x):  
        print(x.shape)
        x = self.linear(x)
        return x

class WhatEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.riesz_layer = RieszLayer(1, 32)  # Assuming input has 1 channel
        self.conv1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        x = self.riesz_layer(x)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# [B, N_ATTR] -> [B, (chan + 1) * obj_size * obj_size]
class ObjectDecoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(ObjectDecoder, self).__init__()
    
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        x = self.linear(x)
        return x

# [B, C, 32, 32] -> [B, 2 * N_ATTR] 
class ObjectConvEncoder(nn.Module):
    
    def __init__(self, input_channels, output_dim):
        super(ObjectConvEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            RieszLayer(input_channels, 16),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        
        self.linear = nn.Linear(256, output_dim)
    
    def forward(self, x):
        
        x = self.cnn(x)
        flat_x = x.flatten(start_dim = 1)
        out = self.linear(flat_x)
        
        return out



# [B, N_ATTR] -> [B, C, 32, 32]
class ObjectConvDecoder(nn.Module):
    
    def __init__(self, out_channels):
        super(ObjectConvDecoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(cfg.n_what, 256),
            nn.ReLU(),
        )
        
        self.cnn = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            # nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.Conv2d(128, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            # nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            # nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            # nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            # nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            # nn.ConvTranspose2d(16, out_channels, 3, 1, 1),
            nn.Conv2d(16, out_channels, 3, 1, 1),
            
            # nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 256, 1, 1)
        out = self.cnn(x)
    
        return out
