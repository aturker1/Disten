from abc import ABC

import numpy as np

from .utils import propagate_mesh
from typing import Callable


class Tensor(ABC):
    ndim: int = 0
    shape: tuple[int, ...]
    mesh_index: tuple[int, ...] | None

    def __init__(
        self, tensor: np.ndarray, mesh_index: tuple[int, ...] | None = None
    ) -> None:
        self.tensor = tensor
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.mesh_index = mesh_index

    def __add__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.tensor + other.tensor)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.tensor - other.tensor)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.tensor * other.tensor)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.tensor / other.tensor)


class ParallelTensor:
    def __init__(
        self, local_tensors: list[Tensor], device_mesh: tuple[int, ...]
    ) -> None:
        self.local_tensors = local_tensors
        self.device_mesh = device_mesh

    def __add__(self, other: "ParallelTensor") -> "ParallelTensor":
        return run_op(self, other, lambda x, y: x + y)

    def __sub__(self, other: "ParallelTensor") -> "ParallelTensor":
        return run_op(self, other, lambda x, y: x - y)

    def __mul__(self, other: "ParallelTensor") -> "ParallelTensor":
        return run_op(self, other, lambda x, y: x * y)

    def __truediv__(self, other: "ParallelTensor") -> "ParallelTensor":
        return run_op(self, other, lambda x, y: x / y)


def run_op(
    tensor1: ParallelTensor, tensor2: ParallelTensor, op: Callable
) -> ParallelTensor:
    assert len(tensor1.local_tensors) == len(
        tensor2.local_tensors
    ), "number of local tensors must match"
    new_mesh = propagate_mesh(tensor1.device_mesh, tensor2.device_mesh)

    new_local_tensors = []
    for local_tensor1, local_tensor2 in zip(
        tensor1.local_tensors, tensor2.local_tensors
    ):
        assert local_tensor1.mesh_index is not None
        assert local_tensor2.mesh_index is not None

        prepared_local_tensor1 = prepare_local_tensor(
            local_tensor1,
            tensor1.device_mesh,
            tensor2.device_mesh,
            local_tensor2.mesh_index,
        )
        prepared_local_tensor2 = prepare_local_tensor(
            local_tensor2,
            tensor2.device_mesh,
            tensor1.device_mesh,
            local_tensor1.mesh_index,
        )
        res = op(prepared_local_tensor1, prepared_local_tensor2)
        new_local_tensors.append(res)

    return ParallelTensor(new_local_tensors, new_mesh)


def prepare_local_tensor(
    local_tensor: Tensor,
    device_mesh: tuple[int, ...],
    target_device_mesh: tuple[int, ...],
    target_mesh_indexes: tuple[int, ...],
) -> Tensor:
    slices = []
    for idx, (local_dim, target_dim, target_mesh_index) in enumerate(
        zip(device_mesh, target_device_mesh, target_mesh_indexes)
    ):
        if local_dim == 1 and target_dim != 1:
            global_shape = local_tensor.tensor.shape[idx]
            local_shape = global_shape // target_dim
            slices.append(
                slice(
                    target_mesh_index * local_shape,
                    (target_mesh_index + 1) * local_shape,
                )
            )
        else:
            slices.append(slice(None))

    return Tensor(local_tensor.tensor[*slices], local_tensor.mesh_index)
