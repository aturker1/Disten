import math

from .tensor import Tensor, ParallelTensor
from .placements import Placement, Shard, Replicate

N_DEVICE = 4


def distribute_tensor(tensor: Tensor, device_mesh: tuple[int]) -> ParallelTensor:
    assert len(device_mesh) == tensor.ndim, "placement length must match tensor dimensions"
    assert N_DEVICE %  math.prod(device_mesh) == 0, "number of devices must be divisible by the product of the mesh"

    local_tensor_shape = []
    indexes = []
    global_tensor_shape = tensor.shape

    for idx, mesh_dim in enumerate(device_mesh):
        if mesh_dim == 1:
            local_tensor_shape.append(global_tensor_shape[idx])
        else:
            local_tensor_shape.append(global_tensor_shape[idx] // mesh_dim)

    
    mesh_indexes = (0,) * len(device_mesh)
    
    local_tensors = []
    for _ in range(N_DEVICE):
        indexes = []

        for dim, mesh_shape in enumerate(device_mesh):
            if mesh_shape == 1:
                indexes.append(slice(0, global_tensor_shape[dim]))
            else:
                indexes.append(slice(mesh_indexes[dim] * local_tensor_shape[dim], (mesh_indexes[dim] + 1) * local_tensor_shape[dim]))


        local_tensors.append(tensor.tensor[*indexes])

        mesh_indexes = _next_mesh_index(mesh_indexes, device_mesh)
        


    parallel_tensor = ParallelTensor(local_tensors=local_tensors, device_mesh=device_mesh)

    return parallel_tensor


# (2,2,2), (2,3,2)
# (0,0), (2,2) -> (0,1)
# (0,1), (2,2) -> (1,0)
# (1,0), (2,2) -> (1,1)

# (2,2), (4,3) -> (3, 0)
# (1,1), (4,3) -> (1, 2)

def _next_mesh_index(mesh_index: tuple[int, ...], device_mesh: tuple[int, ...]):
    current_idx = 0
    prod = 1
    for idx in range(len(device_mesh)-1, -1, -1):
        current_idx += mesh_index[idx] * prod
        prod *= device_mesh[idx]

    next_index = current_idx + 1

    new_mesh_index = []
    for dim_size in reversed(device_mesh):
        new_mesh_index.append(next_index % dim_size)
        next_index //= dim_size
    return tuple(reversed(new_mesh_index))