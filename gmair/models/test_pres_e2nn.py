import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn

class GroupEquivariantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_rotations=8, stride=1, padding=0):
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=num_rotations)
        
        if out_channels % num_rotations != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_rotations ({num_rotations})")

        reduced_in_channels = in_channels // num_rotations
        reduced_out_channels = out_channels // num_rotations
        
        self.in_type = e2nn.FieldType(self.r2_act, reduced_in_channels * [self.r2_act.regular_repr])
        self.out_type = e2nn.FieldType(self.r2_act, reduced_out_channels * [self.r2_act.regular_repr])
        
        self.conv = e2nn.R2Conv(self.in_type, self.out_type, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.in_type)
        x = self.conv(x)
        return x.tensor

class StandardConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

class RotationInvariantGridNetwork(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, topology, num_rotations=8):
        super().__init__()
        self.num_rotations = num_rotations
        self.topology = topology
        self.net = self._build_backbone(n_in_channels, n_out_channels)

    def _build_backbone(self, n_in_channels, n_out_channels):
        n_prev = n_in_channels
        net = nn.ModuleDict()

        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev
            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f
            else:
                f = layer['out_channels']

            if f % self.num_rotations != 0:
                raise ValueError(f"Layer {i}: out_channels ({f}) must be divisible by num_rotations ({self.num_rotations})")

            net[f'rot_conv_{i}'] = GroupEquivariantConv2d(**layer, num_rotations=self.num_rotations)
            net[f'act_{i}'] = nn.ReLU()

            if i < len(self.topology) - 1:
                net[f'pool_{i}'] = nn.MaxPool2d(kernel_size=2, stride=2)

            n_prev = f

        # Ensure the final output convolution produces the correct output size
        net['conv_out'] = StandardConv2d(
            in_channels=f, 
            out_channels=n_out_channels, 
            kernel_size=1, 
            stride=1
        )
        
        return net

    def forward(self, x):
        for name, layer in self.net.items():
            x = layer(x)
        return x

def get_heat_map_head(cfg):
    n_features = cfg.n_backbone_features  # Number of input features (from the backbone)
    n_class = 1  # Since we are predicting binary class (object/no object in grid)
    
    # Topology defining the convolutional layers
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (128x128)
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (128x128)
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (128x128)
    ]
    
    # Create the rotation-invariant grid network using group convolutions
    model = RotationInvariantGridNetwork(n_features, n_class, topology)
    
    return model

# Example configuration object
class Config:
    n_backbone_features = 256  # Number of features from backbone

# Example usage
cfg = Config()
heatmap_head = get_heat_map_head(cfg)

# Example input (batch_size=1, channels=256, height=128, width=128)
x = torch.randn(1, 256, 128, 128)

# Get the output heatmap (logits)
output = heatmap_head(x)

print(output.shape)  # Check output shape

