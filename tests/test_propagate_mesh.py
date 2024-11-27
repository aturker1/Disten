from src.utils import propagate_mesh
import pytest


def test_propagate_same_mesh():
    mesh1 = (2, 2)
    mesh2 = (2, 2)
    new_mesh = propagate_mesh(mesh1, mesh2)
    assert new_mesh == (2, 2)


def test_propagate_different_mesh():
    mesh1 = (2, 2)
    mesh2 = (1, 2)
    new_mesh = propagate_mesh(mesh1, mesh2)
    assert new_mesh == (2, 2)


def test_propagate_uncompatible():
    mesh1 = (2, 2)
    mesh2 = (2, 3)

    with pytest.raises(ValueError) as e:
        propagate_mesh(mesh1, mesh2)

    assert str(e.value) == "Meshes are not compatible"


def test_propagete_different_length():
    mesh1 = (2, 2)
    mesh2 = (2, 2, 2)

    with pytest.raises(ValueError) as e:
        propagate_mesh(mesh1, mesh2)

    assert str(e.value) == "Meshes must have the same length"
