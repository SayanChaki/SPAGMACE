from collections import OrderedDict
from e2cnn import gspaces
from e2cnn.nn import FieldType, R2Conv, InnerBatchNorm, ReLU, ELU, GroupPooling
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, Linear, Module, ModuleDict, ModuleList
from torch.nn import functional as F

from gmair.config import config as cfg
from gmair.utils import debug_tools
from torch.nn import Module, Sequential, ReLU
from collections import OrderedDict
from e2cnn import gspaces
from e2cnn import nn as e2nn
import torch

'''class BasicNetwork(Module):
    def __init__(self, n_in_channels, n_out_channels, topology, internal_activation=ReLU):

        super().__init__()
        self.topology = topology
        
        self.net = self._build_backbone(n_in_channels, n_out_channels)
    def _build_backbone(self, n_in_channels, n_out_channels):

    
     print("Initializing backbone with {} input channels and {} output channels.".format(n_in_channels, n_out_channels))
    
     # Define a symmetry group (e.g., rotation group for 2D)
     self.gspace = gspaces.Rot2dOnR2(8)  # 8-fold rotational symmetry
     print("Symmetry group defined:", self.gspace)
    
     # Define the input and output field types
     c_in = FieldType(self.gspace, [self.gspace.trivial_repr] * n_in_channels)
     c_hid = FieldType(self.gspace, [self.gspace.regular_repr] * 3)  # Adjust based on your topology
     c_out = FieldType(self.gspace, [self.gspace.regular_repr] * n_out_channels)
     print("Field types set: c_in =", c_in, "c_hid =", c_hid, "c_out =", c_out)

     # Create an ordered dict to build the network layers
     net = OrderedDict()
     print("OrderedDict for network layers initialized.")

     # Build the layers according to your topology
     for i, layer in enumerate(self.topology):
        print(f"Building layer {i} with topology:", layer)
        
        if i == 0:
            # First layer using R2Conv with input type
            net[f'conv_{i}'] = R2Conv(c_in, c_hid, kernel_size=5, bias=False)
            print(f"R2Conv layer {i} created with input type c_in and output type c_hid")
        else:
            # Use R2Conv for subsequent layers
            net[f'conv_{i}'] = R2Conv(c_hid, c_hid, kernel_size=3, bias=False)
            print(f"R2Conv layer {i} created with input type c_hid and output type c_hid")
        
        net[f'bn_{i}'] = InnerBatchNorm(c_hid)
        print(f"InnerBatchNorm layer {i} created for c_hid")

        net[f'act_{i}'] = ReLU(c_hid, inplace=True)
        print(f"ReLU activation layer {i} created for c_hid")

        # Add a pooling layer if necessary
        if i == len(self.topology) - 1:  # Last layer
            net[f'pool_{i}'] = GroupPooling(c_hid)
            print(f"GroupPooling layer {i} created for c_hid")
    
     # Add the final convolution layer
     net[f'conv_out'] = R2Conv(c_hid, c_out, kernel_size=3, bias=False)
     print("Final R2Conv layer created with input type c_hid and output type c_out")

     net[f'bn_out'] = InnerBatchNorm(c_out)
     print("Final InnerBatchNorm layer created for c_out")

     net[f'act_out'] = ELU(c_out, inplace=True)
     print("Final ELU activation layer created for c_out")

     # Create the SequentialModule
     print("Returning nn.Sequential with all layers.")
     return nn.Sequential(*net.values())

    def forward(self, x):
        print(x.shape)
        x = self.net(x)
        return x'''


"""from collections import OrderedDict
from torch.nn import Module
from e2cnn import nn as ecnn, gspaces

class BasicNetwork(Module):
    def __init__(self, n_in_channels, n_out_channels, topology, symmetry=2, internal_activation=ecnn.ReLU):
        super().__init__()
        
        # Define the symmetry (e.g., rotational symmetry)
        self.symmetry = symmetry
        self.r2_act = gspaces.Rot2dOnR2(N=symmetry)  # e.g., N=8 for 8-fold rotational symmetry
        
        self.topology = topology
        self.net = self._build_backbone(n_in_channels, n_out_channels, internal_activation)

    def _build_backbone(self, n_in_channels, n_out_channels, internal_activation):
        
        n_prev = n_in_channels
        net = OrderedDict()
        
        # Initial feature type based on the number of input channels
        feat_type_in = ecnn.FieldType(self.r2_act, n_prev * [self.r2_act.trivial_repr])

        # Builds internal layers except for the last layer
        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev
            
            # Update output channels and feature type for each layer
            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f
            else:
                f = layer['out_channels']
            
            feat_type_out = ecnn.FieldType(self.r2_act, f * [self.r2_act.regular_repr])

            # Debugging: Print the layer parameters before using them
            print(f"Layer {i} parameters:", layer)

            # Keep only allowed arguments for R2Conv
            allowed_keys = {'kernel_size', 'stride', 'padding', 'dilation'}
            layer = {k: layer[k] for k in layer if k in allowed_keys}

            # Use R2Conv from e2cnn for equivariant convolution
            net[f'conv_{i}'] = ecnn.R2Conv(feat_type_in, feat_type_out, **layer)
            net[f'act_{i}'] = internal_activation(feat_type_out)

            # Update the input feature type for the next layer
            feat_type_in = feat_type_out
            n_prev = f
        
        # Builds the final layer with output channels
        feat_type_out = ecnn.FieldType(self.r2_act, n_out_channels * [self.r2_act.trivial_repr])
        net['conv_out'] = ecnn.R2Conv(feat_type_in, feat_type_out, kernel_size=1, stride=1)
        
        return ecnn.SequentialModule(*net.values())

    def forward(self, x):
        # Ensure x is a GeometricTensor
        if not isinstance(x, ecnn.GeometricTensor):
            raise TypeError("Input must be a GeometricTensor with the appropriate feature type.")
        
        # Pass the input through the network
        x = self.net(x)
        return x"""



class BasicNetwork(Module):
    def __init__(self, n_in_channels, n_out_channels, topology, internal_activation=ReLU):

        super().__init__()
        self.topology = topology
        
        self.net = self._build_backbone(n_in_channels, n_out_channels)

    def _build_backbone(self, n_in_channels, n_out_channels):


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
        print("BEfore basic block")
        print(x.shape)
        x = self.net(x)
        print("AFTER BASIC BLOXk")
        print(x.shape)
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
        x = self.linear(x)
        return x


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
'''
# [B, C, 32, 32] -> [B, 2 * N_ATTR] 
class ObjectConvEncoder(nn.Module):
    
    def __init__(self, input_channels):
        super(ObjectConvEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 1, 1),
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
        
        self.linear = nn.Linear(256, 2 * cfg.N_ATTRIBUTES)
    
    def forward(self, x):
        
        x = self.cnn(x)
        flat_x = x.flatten(start_dim = 1)
        out = self.linear(flat_x)
        
        return out

'''

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
