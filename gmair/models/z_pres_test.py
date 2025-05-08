from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear, Module, ModuleDict, ModuleList
from torch.nn import functional as F

from gmair.config import config as cfg
from gmair.utils import debug_tools

class RieszTransform(nn.Module):
    def __init__(self):
        super(RieszTransform, self).__init__()

    def forward(self, x):
        def FT(I):
            return torch.fft.fftshift(torch.fft.fft2(I))
        
        def iFT(I):
            return torch.fft.ifft2(torch.fft.ifftshift(I))

        def riesz_transform(image_tensor):
            batch_size, _, rows, cols = image_tensor.shape

            # Create Riesz kernels
            u = torch.arange(0, rows, dtype=torch.float32, device=image_tensor.device).reshape(-1, 1) - rows / 2
            v = torch.arange(0, cols, dtype=torch.float32, device=image_tensor.device).reshape(1, -1) - cols / 2
            u = u / rows
            v = v / cols

            R1 = -1j * u / torch.sqrt(u**2 + v**2)
            R2 = -1j * v / torch.sqrt(u**2 + v**2)

            # Avoid division by zero at the (0,0) frequency
            R1[rows // 2, cols // 2] = 0
            R2[rows // 2, cols // 2] = 0

            # Discrete Fourier transform of I
            I_hat = FT(image_tensor)

            # First-order Riesz transform
            I1 = iFT(I_hat * R1).real
            I2 = iFT(I_hat * R2).real

            # Second-order Riesz transform
            I_20 = iFT(I_hat * R1 * R1).real
            I_02 = iFT(I_hat * R2 * R2).real
            I_11 = iFT(I_hat * R1 * R2).real

            return torch.stack([I1, I2, I_20, I_02, I_11], dim=1)

        # Apply Riesz transform to each channel
        L_R = []
        for i in range(x.shape[1]):
            transformed = riesz_transform(x[:, i, :, :].unsqueeze(1))
            L_R.append(transformed)
        
        L_R = torch.cat(L_R, dim=1)
        L_R = L_R.squeeze(2)
        return L_R

class RieszLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RieszLayer, self).__init__()
        self.riesz = RieszTransform()
        self.conv = nn.Conv2d(in_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.riesz(x)
        x = self.conv(x)
        return x
        
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
        
def get_heat_map_head():
    n_features = cfg.n_backbone_features
    
    n_class = 1
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_class, topology)
    
def get_where_head():
    n_features = cfg.n_backbone_features
    
    n_localization_latent = 8  # mean and var for (y, x, h, w)
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_localization_latent, topology)
        
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
