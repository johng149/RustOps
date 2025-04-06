import torch
from torch import Tensor
import numpy as np
import os

def save_reference(
    x: Tensor,
    dir: str,
    name: str,
):
    """
    Save given tensor as npz file
    Args:
        x (Tensor): Tensor to save
        dir (str): Directory to save the tensor
        name (str): Name of the file to save the tensor
    """
    # Ensure directory and its parent directories exist
    os.makedirs(dir, exist_ok=True)
    # Convert tensor to numpy array
    x_np = x.detach().cpu().numpy()
    # Save numpy array as npz file
    np.savez_compressed(os.path.join(dir, name), x=x_np)