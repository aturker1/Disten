from abc import ABC

import numpy as np

from .placements import Placement


class Tensor(ABC):
    def __init__(self, tensor: np.ndarray) -> None:
        self.tensor = tensor
        self.shape = tensor.shape
        self.ndim = tensor.ndim

    ndim: int = 0
    shape: tuple[int] = ()

    


class ParallelTensor:
    def __init__(self, local_tensors: list[Tensor], device_mesh: tuple[int]) -> None:
        self.local_tensors = local_tensors
        self.device_mesh = device_mesh

    def __add__(self, other: 'ParallelTensor') -> 'ParallelTensor':
        assert self.device_mesh == other.device_mesh, "device mesh must match"
        assert len(self.local_tensors) == len(other.local_tensors), "number of local tensors must match"

        new_local_tensors = []
        for local_tensor1, local_tensor2 in zip(self.local_tensors, other.local_tensors):
            new_local_tensors.append(Tensor(local_tensor1.tensor + local_tensor2.tensor))

        return ParallelTensor(new_local_tensors, self.device_mesh)