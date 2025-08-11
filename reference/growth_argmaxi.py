import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
from maxi import create_tensors as create_tensors_maxi
from argmaxi import create_tensors as create_tensors_argmaxi
from mark_reserved_indices import create_tensors as create_tensors_mark_reserved_indices
from move_value_to_back import create_tensors as create_tensors_move_value_to_back
from expand_for_batches import create_tensors as create_tensors_expand_for_batches
import einops

def create_tensors(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        eps = 1e-8,
        threshold = 0.5,
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "growth_argmaxi",
        x = None,
        counts = None
):
    assert len(shape) == 3, "Shape must be of length 3"
    x = torch.rand(shape, dtype=dtype) if x is None else x
    counts = torch.randint(0, 4, shape[1:], dtype=torch.float32) if counts is None else counts
    
    batch_size, nodes, mems = x.shape

    _, normal_path = create_tensors_argmaxi(shape=None, x=x)
    trigger_growth = torch.sum(x > threshold, dim=-1, keepdim=True) <= 0
    mark = -2
    avail, all, _, _, _ = create_tensors_mark_reserved_indices(shape=None, mark=mark, acts=normal_path, counts=counts, trigger=trigger_growth)
    avail, _ = create_tensors_move_value_to_back(shape=None, x=avail)
    avail, _ = create_tensors_expand_for_batches(shape=None, x=avail, batch_size=batch_size)
    all, _ = create_tensors_expand_for_batches(shape=None, x=all, batch_size=batch_size)
    final = torch.where(avail == mark, all, avail)
    indices_sg = final.transpose(-1, -2).reshape(batch_size, -1, 1)
    growth_path = torch.zeros_like(normal_path)
    values_of_interest = torch.gather(x, -1, indices_sg)
    values_of_interest[values_of_interest == 0] = 1
    growth_path = torch.scatter(growth_path, -1, indices_sg, values_of_interest)
    growth_path = torch.abs(growth_path)
    _, growth_path = create_tensors_argmaxi(shape=None, x=growth_path)
    grown = torch.where(trigger_growth, growth_path, normal_path)

    updated_counts = einops.reduce(grown, 'batch nodes mems -> nodes mems', 'sum')
    updated_counts = torch.where(updated_counts <= 0, counts, updated_counts)

    return grown, updated_counts, x, counts

def create_growth_argmaxi(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        eps = 1e-8,
        threshold = 0.5,
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "growth_argmaxi",
):
    grown, updated_counts, x, counts = create_tensors(shape, eps, threshold, dtype, dir, name)
    save_reference(grown, dir, f"{name}_grown")
    save_reference(updated_counts, dir, f"{name}_updated_counts")
    save_reference(x, dir, f"{name}_x")
    save_reference(counts, dir, f"{name}_counts")

if __name__ == "__main__":
    create_growth_argmaxi((6, 7, 8), eps=1e-8, threshold=0.5, dtype=torch.float32, dir="data", name="growth_argmaxi")