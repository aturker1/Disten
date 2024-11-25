

def propagate_mesh(left_mesh: tuple[int, ...], right_mesh: tuple[int, ...]):
    if len(left_mesh) != len(right_mesh):
        raise ValueError("Meshes must have the same length")
    
    new_mesh = []
    for left_dim, right_dim in zip(left_mesh, right_mesh):
        if left_dim == 1:
            new_mesh.append(right_dim)
        elif right_dim == 1:
            new_mesh.append(left_dim)
        else:
            if left_dim != right_dim:
                raise ValueError("Meshes are not compatible")
            new_mesh.append(left_dim)
    
    return tuple(new_mesh)