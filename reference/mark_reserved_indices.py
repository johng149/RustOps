import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
import einops

def create_tensors(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        mark = -2,
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "mark_reserved_indices",
        acts = None,
        counts = None,
        trigger = None
):
    x = torch.rand(shape, dtype=dtype) if acts is None else acts
    y = torch.randint(0, 4, shape[1:], dtype=torch.float32) if counts is None else counts
    z = torch.randn(list(shape[:2]) + [1], dtype=dtype) > 0.5 if trigger is None else trigger
    used_values, used_indices = torch.sort(x, dim=-1)
    reservations = used_indices[:,:,-1:]
    reservations[z] = -1
    reservations = einops.rearrange(reservations, 'batch mems flag -> mems (batch flag)')
    sorts, sort_indices = torch.sort(y, dim=-1)
    sorts, sort_indices, reservations
    reservation_mask = reservations != -1
    expanded_usages = sort_indices.unsqueeze(1).expand(-1, reservations.size(1), -1)
    expanded_res = reservations.unsqueeze(-1).expand(-1, -1, y.size(1))
    matches = (expanded_res == expanded_usages) & reservation_mask.unsqueeze(-1)
    matches = matches.any(dim=1)
    return torch.where(matches, mark, sort_indices), sort_indices, x, y, z

def create_mark_reserved_indices(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        mark = -2,
        dtype: torch.dtype = torch.float32,
        dir: str = "data",
        name: str = "mark_reserved_indices",
):
    avail, all, normal, counts, trigger = create_tensors(shape, mark, dtype, dir, name)
    save_reference(avail, dir, f"{name}_avail")
    save_reference(all, dir, f"{name}_all")
    save_reference(normal, dir, f"{name}_normal")
    save_reference(counts, dir, f"{name}_counts")
    save_reference(trigger, dir, f"{name}_trigger")

if __name__ == "__main__":
    # batch size, nodes, mems
    create_mark_reserved_indices((6, 7, 8), mark=-2, dtype=torch.float32, dir="data", name="mark_reserved_indices")