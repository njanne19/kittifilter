import torch
import torch.nn as nn
from fannypack.nn import resblocks

state_dim = 2
control_dim = 13
obs_gps_dim = 2

def state_layers(units: int) -> nn.Module:
    """Create a state encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(state_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


def control_layers(units: int) -> nn.Module:
    """Create a control command encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(control_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


def observation_image_layers(units: int) -> nn.Module:
    """Create an raw image + difference image  encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        resblocks.Conv2d(channels=32, kernel_size=3),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
        nn.Flatten(),  # 32 * 32 * 8
        nn.Linear(8 * 124 * 409, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )

def observation_gps_layers(units: int) -> nn.Module: 
    """Create a GPS observation encoder block.
    
    Args: 
        units (int): # of hidden units in network layers.
        
    Returns: 
        nn.Module: Encoder block. 
    """
    
    return nn.Sequential(
        nn.Linear(obs_gps_dim, units), 
        nn.ReLU(inplace=True),
        resblocks.Linear(units),    
    )
