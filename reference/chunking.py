import torch
from util.save_reference import save_reference
from typing import Tuple

def chunking(tensor: torch.Tensor, chunk_height: int, chunk_width: int, pad: int = 0) -> torch.Tensor:
    """
    Reshapes a 4D tensor (batch, height, width, channels) into chunks.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch, height, width, channels).
        chunk_height (int): The height of each chunk.
        chunk_width (int): The width of each chunk.
        pad (int): Value used for padding if the dimensions are not perfectly divisible
                   by chunk_height/chunk_width. Defaults to 0.

    Returns:
        torch.Tensor: A 3D tensor of shape (batch, num_total_chunks, chunk_height * chunk_width).
                      Chunks are ordered by width_step, then height_step, then channel.
    """
    batch, height, width, channels = tensor.shape

    if chunk_height <= 0 or chunk_width <= 0:
        raise ValueError("chunk_height and chunk_width must be positive.")

    num_h_steps = (height + chunk_height - 1) // chunk_height
    height_padded = num_h_steps * chunk_height
    num_w_steps = (width + chunk_width - 1) // chunk_width
    width_padded = num_w_steps * chunk_width

    height_needed_pad = height_padded - height
    width_needed_pad = width_padded - width

    # Padding format for torch.nn.functional.pad is (pad_left_dimN, pad_right_dimN, ...).
    # For a 4D tensor (B, H, W, C), tensor.pad expects padding for last 3 dims (C, W, H)
    # So, padding tuple is (pad_C_left, pad_C_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right)
    padding = (0, 0,  # Channels padding (none)
               0, width_needed_pad,   # Width padding (only on the right)
               0, height_needed_pad)  # Height padding (only at the bottom)

    padded_tensor = torch.nn.functional.pad(tensor, padding, mode="constant", value=pad)

    # Reshape to (batch, num_h_steps, chunk_height, num_w_steps, chunk_width, channels)
    reshaped = padded_tensor.reshape(batch, num_h_steps, chunk_height, num_w_steps, chunk_width, channels)

    # Permute to (batch, num_w_steps, num_h_steps, channels, chunk_height, chunk_width)
    # This order ensures that when we flatten, chunks are grouped by w_step, then h_step, then channel.
    # Original indices: (0:B, 1:N_h, 2:C_h, 3:N_w, 4:C_w, 5:Chan)
    # New order:        (0:B, 3:N_w, 1:N_h, 5:Chan, 2:C_h, 4:C_w)
    permuted = reshaped.permute(0, 3, 1, 5, 2, 4)

    # Reshape to (batch, num_total_chunks, chunk_area)
    # num_total_chunks = num_w_steps * num_h_steps * channels
    # chunk_area = chunk_height * chunk_width
    output_tensor = permuted.reshape(batch, num_w_steps * num_h_steps * channels, chunk_height * chunk_width)

    return output_tensor

def create_chunking(
    shape: Tuple[int, int, int, int],
    chunk_height: int,
    chunk_width: int,
    pad_value: int = 0,
    dtype: torch.dtype = torch.float32,
    dir: str = "data",
    name: str = "chunking",
):
    """
    Create a tensor, perform chunking, and save the original and chunked tensors.

    Args:
        shape (Tuple[int, int, int, int]): Shape of the tensor (batch, height, width, channels).
        chunk_height (int): Height of each chunk.
        chunk_width (int): Width of each chunk.
        pad_value (int): Value to use for padding. Default is 0.
        dtype (torch.dtype): Data type of the tensor. Default is torch.float32.
        dir (str): Directory to save the reference tensors. Default is "data".
        name (str): Base name for the reference tensor files. Default is "chunking".
    """
    if len(shape) != 4:
        raise ValueError("Input shape must be 4D (batch, height, width, channels).")
    if any(s <= 0 for s in shape):
        raise ValueError("All dimensions in shape must be positive.")
    x = torch.randn(shape, dtype=dtype)


    save_reference(x, dir, f"{name}_x")

    chunked_tensor = chunking(x, chunk_height, chunk_width, pad=pad_value)
    save_reference(chunked_tensor, dir, f"{name}_result_ch{chunk_height}_cw{chunk_width}_pad{pad_value}")

if __name__ == "__main__":
    create_chunking(
        shape=(2, 32, 32, 3),  # batch, height, width, channels
        chunk_height=8,
        chunk_width=8,
        pad_value=0,
        dtype=torch.float32,
        dir="data",
        name="chunking_32x32_ch8cw8_float"
    )

    create_chunking(
        shape=(1, 5, 7, 1),   # batch, height, width, channels
        chunk_height=2,
        chunk_width=3,
        pad_value=0,          # Pad with zero
        dtype=torch.float32,
        dir="data",
        name="chunking_5x7_ch2cw3_int_pad0"
    )

    create_chunking(
        shape=(1, 3, 3, 2),   # batch, height, width, channels
        chunk_height=2,
        chunk_width=2,
        pad_value=5,          # Pad with five
        dtype=torch.float32,
        dir="data",
        name="chunking_3x3_ch2cw2_float16_pad5"
    )

    # Example 4: Single large chunk (entire feature map)
    create_chunking(
        shape=(1, 7, 7, 1),
        chunk_height=7,
        chunk_width=7,
        pad_value=0,
        dtype=torch.float32,
        dir="data",
        name="chunking_7x7_ch7cw7_full"
    )

    # Example 5: 1x1 chunks (effectively a flatten operation with channel interleaving)
    create_chunking(
        shape=(1, 2, 2, 2),
        chunk_height=1,
        chunk_width=1,
        pad_value=0,
        dtype=torch.float32,
        dir="data",
        name="chunking_2x2_ch1cw1_int8"
    )
    print("Chunking reference tensors created in 'data' directory.")
