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
        # Get filter rotations
        rotated_weights = self.get_rotated_weights(self.weight)

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

    def rotate_filter(self, filter, angle):
        # Rotate the filter by a given angle
        theta = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float)

        grid = F.affine_grid(theta.unsqueeze(0), filter.unsqueeze(0).size(), align_corners=False)
        rotated_filter = F.grid_sample(filter.unsqueeze(0), grid, align_corners=False)
        return rotated_filter.squeeze(0)

    def get_rotated_weights(self, weights):
        # Rotate the convolution filters by different angles
        rotated_weights = []
        for i in range(self.num_rotations):
            angle = (2 * math.pi * i) / self.num_rotations
            rotated_filters = []
            for j in range(weights.size(0)):  # for each output channel
                rotated_filter = self.rotate_filter(weights[j], angle)
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
            net[f'act_{i}'] = nn.ReLU()  # Using ReLU as activation for intermediate layers

            # Add a max-pooling layer after some layers to downsample
            if i < len(self.topology) - 1:  # Add pooling before the last layer
                net[f'pool_{i}'] = nn.MaxPool2d(kernel_size=2, stride=2)

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
        return x

def get_heat_map_head(cfg):
    n_features = cfg.n_backbone_features  # Number of input features (from the backbone)
    n_class = 1  # Since we are predicting binary class (object/no object in grid)
    
    # Topology defining the convolutional layers
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (64x64)
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (64x64)
        dict(filters=128, kernel_size=3, stride=1, padding=1),  # Keep spatial size (64x64)
    ]
    
    # Create the rotation-invariant grid network
    model = RotationInvariantGridNetwork(n_features, n_class, topology)
    
    return model

# Example configuration object
class Config:
    n_backbone_features = 256  # Number of features from backbone

# Example usage
cfg = Config()
heatmap_head = get_heat_map_head(cfg)

# Example input (batch_size=1, channels=256, height=64, width=64)
x = torch.randn(1, 256, 128, 128)

# Get the output heatmap (logits)
output = heatmap_head(x)

print(output.shape)  # Expected output shape: [1, 1, 16, 16]

