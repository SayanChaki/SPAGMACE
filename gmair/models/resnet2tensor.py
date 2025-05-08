import torch
from e2cnn import gspaces
from e2cnn import nn as e2nn
from mmrotate.models.backbones import ReResNet
import pytest


from mmrotate.models.backbones import ReResNet
def convert_reresnet_output(geometric_tensor: e2nn.GeometricTensor) -> torch.Tensor:
    """
    Convert ReResNet's output geometric tensor to standard PyTorch tensor.
    Handles the specific case of output shape [1,64,8,8] with 8 rotation fields.
    
    Args:
        geometric_tensor: Output from ReResNet with field type 
                        [8-Rotations: {regular x 8}] and shape [1,64,8,8]
        
    Returns:
        torch.Tensor: Standard PyTorch tensor with channels reorganized
    """
    # Get the underlying tensor
    tensor = geometric_tensor.tensor
    
    # For ReResNet output: [1,64,8,8]
    batch_size = tensor.shape[0]  # 1
    channels = tensor.shape[1]    # 64
    height = tensor.shape[2]      # 8
    width = tensor.shape[3]       # 8
    
    # The number of actual feature channels is channels // 8
    # since each feature has 8 rotational copies
    features_per_rotation = channels // 8  # 64 // 8 = 8
    
    # Reshape to separate rotation and feature dimensions
    # [B, C, H, W] -> [B, num_features, num_rotations, H, W]
    reshaped = tensor.view(batch_size, features_per_rotation, 8, height, width)
    
    # If you want to combine features and rotations into a single channel dimension:
    # [B, num_features, num_rotations, H, W] -> [B, num_features * num_rotations, H, W]
    final_tensor = reshaped.reshape(batch_size, channels, height, width)
    
    return final_tensor

def convert_to_reresnet_geometric(
    standard_tensor: torch.Tensor,
    gspace: gspaces.GSpace = None
) -> e2nn.GeometricTensor:
    """
    Convert a standard PyTorch tensor back to ReResNet's geometric tensor format.
    
    Args:
        standard_tensor: PyTorch tensor of shape [1,64,8,8]
        gspace: Optional G-space for 8 rotations. If None, creates new one.
        
    Returns:
        e2nn.GeometricTensor: Geometric tensor compatible with ReResNet
    """
    if gspace is None:
        # Create a G-space with 8 rotations (2π/8 = 45 degrees)
        gspace = gspaces.Rotation2dOnR2(N=8)
    
    # Create the regular representation
    reg_type = gspace.regular_repr
    
    # Create field type with 8 regular fields for each feature
    num_features = standard_tensor.shape[1] // 8
    field_type = e2nn.FieldType(gspace, [reg_type] * num_features)
    
    # Create geometric tensor
    return e2nn.GeometricTensor(standard_tensor, field_type)


def example_reresnet(feat):
    # Initialize ReResNet
    '''model = ReResNet(
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch')'''
    
    # Create rotation group space
    gspace = gspaces.Rot2dOnR2(8)
    
    # Forward pass
    #imgs = torch.randn(1, 3, 128, 128)
    #geometric_output = model(imgs)[0]  # Get first output
    geometric_output=feat
    # Convert to standard tensor
    standard_tensor = convert_reresnet_output(geometric_output)
    
    # Convert back if needed
    geometric_tensor = convert_to_reresnet_geometric(standard_tensor, gspace)
    
    return standard_tensor

'''std, geo = example_with_reresnet()
print(std.type)
print(geo.type)'''
