import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from maxi import create_tensors as create_tensors_maxi

def create_tensors(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        eps = 1e-8,
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "argmaxi",
        x = None
):
     x, max_indices, blank, gathered, scattered = create_tensors_maxi(shape, dtype, x=x)
     maxied = torch.abs(scattered)
     factors, _ = torch.max(maxied, dim=-1)
     factors = factors - eps
     result = maxied / factors.unsqueeze(-1)
     return x, result

def create_argmaxi(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        eps = 1e-8,
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "argmaxi",
):
    x, result = create_tensors(shape, eps, dtype)
    save_reference(x, dir, f"{name}_x")
    save_reference(result, dir, f"{name}_result_{eps}")

if __name__ == "__main__":
    # batch size, nodes, mems
    create_argmaxi((6, 7, 8), eps=1e-8, dtype=torch.float32, dir="data", name="argmaxi")