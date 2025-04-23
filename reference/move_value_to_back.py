import torch
from util.save_reference import save_reference
from typing import Tuple, List, Iterable
import einops

def create_tensors(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        value = -2,
        dtype: torch.dtype = torch.int64,
        x = None,
):
    assert  (shape is None and x is not None) or len(shape) == 2, "Shape must be 2D"
    if x is None:
        x = torch.randint(0, 4, shape, dtype=dtype)
        # Randomly make some values equal to `value`
        mask = torch.rand(shape) < 0.2  # 20% of values will be set to `value`
        x[mask] = value

    indices = (x == value).argsort(dim=-1, stable=True)
    return torch.gather(x, -1, indices), x

def create_move_value_to_back(
        shape: Tuple[int, ...] | List[int] | Iterable[int],
        value = -2,
        dtype: torch.dtype = torch.int64,
        dir: str = "data",
        name: str = "move_value_to_back",
):
    moved, original = create_tensors(shape, value, dtype)
    save_reference(moved, dir, f"{name}_moved")
    save_reference(original, dir, f"{name}_original")


if __name__ == "__main__":
    # batch size, nodes
    create_move_value_to_back((14, 17), value=-2, dtype=torch.int64, dir="data", name="move_value_to_back")