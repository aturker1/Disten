import math

from .tensor import Tensor, ParallelTensor

N_DEVICE = 4


def distribute_tensor(tensor: Tensor, device_mesh: tuple[int, ...]) -> ParallelTensor:
    # Validate input types
    if not isinstance(device_mesh, tuple):
        raise ValueError(f"device_mesh must be a tuple, got {type(device_mesh)}")

    # Validate tensor dimensions match mesh
    if len(device_mesh) != tensor.ndim:
        raise ValueError(
            f"device_mesh dimensions ({len(device_mesh)}) must match tensor dimensions ({tensor.ndim})"
        )

    # Validate device count
    mesh_size = math.prod(device_mesh)
    if N_DEVICE % mesh_size != 0:
        raise ValueError(
            f"Number of devices ({N_DEVICE}) must be divisible by product of device_mesh ({mesh_size})"
        )

    # Validate mesh dimensions
    if not all(isinstance(d, int) and d > 0 for d in device_mesh):
        raise ValueError("All device_mesh dimensions must be positive integers")

    local_tensor_shape = []
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
                indexes.append(
                    slice(
                        mesh_indexes[dim] * local_tensor_shape[dim],
                        (mesh_indexes[dim] + 1) * local_tensor_shape[dim],
                    )
                )

        local_tensors.append(Tensor(tensor.tensor[*indexes], mesh_indexes))

        mesh_indexes = _next_mesh_index(mesh_indexes, device_mesh)

    parallel_tensor = ParallelTensor(
        local_tensors=local_tensors, device_mesh=device_mesh
    )

    return parallel_tensor


def _next_mesh_index(mesh_index: tuple[int, ...], device_mesh: tuple[int, ...]):
    current_idx = 0
    prod = 1
    for idx in range(len(device_mesh) - 1, -1, -1):
        current_idx += mesh_index[idx] * prod
        prod *= device_mesh[idx]

    next_index = current_idx + 1

    new_mesh_index = []
    for dim_size in reversed(device_mesh):
        new_mesh_index.append(next_index % dim_size)
        next_index //= dim_size
    return tuple(reversed(new_mesh_index))
