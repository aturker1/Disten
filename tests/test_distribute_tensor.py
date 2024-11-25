import numpy as np
from src.tensor import Tensor
from src.parallel import distribute_tensor
from src import parallel


def test_shard_one_dim():
    inp = np.random.rand(16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (4,))

    assert parallel_tensor.device_mesh == (4,)
    assert len(parallel_tensor.local_tensors) == 4

    for i in range(4):
        assert np.array_equal(
            parallel_tensor.local_tensors[i].tensor, inp[i * 4 : i * 4 + 4]
        )


def test_shard_two_dim():
    inp = np.random.rand(4, 4)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (2, 2))

    assert parallel_tensor.device_mesh == (2, 2)
    assert len(parallel_tensor.local_tensors) == 4

    assert np.array_equal(parallel_tensor.local_tensors[0].tensor, inp[0:2, 0:2])
    assert np.array_equal(parallel_tensor.local_tensors[1].tensor, inp[0:2, 2:4])
    assert np.array_equal(parallel_tensor.local_tensors[2].tensor, inp[2:4, 0:2])
    assert np.array_equal(parallel_tensor.local_tensors[3].tensor, inp[2:4, 2:4])


def test_shard_three_dim():
    inp = np.random.rand(4, 4, 6)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 8
    parallel_tensor = distribute_tensor(tensor, (2, 2, 2))

    assert parallel_tensor.device_mesh == (2, 2, 2)
    assert len(parallel_tensor.local_tensors) == 8

    assert np.array_equal(parallel_tensor.local_tensors[0].tensor, inp[0:2, 0:2, 0:3])
    assert np.array_equal(parallel_tensor.local_tensors[1].tensor, inp[0:2, 0:2, 3:6])
    assert np.array_equal(parallel_tensor.local_tensors[2].tensor, inp[0:2, 2:4, 0:3])
    assert np.array_equal(parallel_tensor.local_tensors[3].tensor, inp[0:2, 2:4, 3:6])
    assert np.array_equal(parallel_tensor.local_tensors[4].tensor, inp[2:4, 0:2, 0:3])
    assert np.array_equal(parallel_tensor.local_tensors[5].tensor, inp[2:4, 0:2, 3:6])
    assert np.array_equal(parallel_tensor.local_tensors[6].tensor, inp[2:4, 2:4, 0:3])
    assert np.array_equal(parallel_tensor.local_tensors[7].tensor, inp[2:4, 2:4, 3:6])


def test_replicate_one_dim():
    inp = np.random.rand(16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (1,))

    assert parallel_tensor.device_mesh == (1,)
    assert len(parallel_tensor.local_tensors) == 4

    for i in range(4):
        assert np.array_equal(parallel_tensor.local_tensors[i].tensor, inp)


def test_replicate_two_dim():
    inp = np.random.rand(16, 16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (1, 1))

    assert parallel_tensor.device_mesh == (1, 1)
    assert len(parallel_tensor.local_tensors) == 4

    for i in range(4):
        assert np.array_equal(parallel_tensor.local_tensors[i].tensor, inp)


def test_shard_replicate_two_dim_1():
    inp = np.random.rand(16, 16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (4, 1))

    assert parallel_tensor.device_mesh == (4, 1)
    assert len(parallel_tensor.local_tensors) == 4

    assert np.array_equal(parallel_tensor.local_tensors[0].tensor, inp[0:4, :])
    assert np.array_equal(parallel_tensor.local_tensors[1].tensor, inp[4:8, :])
    assert np.array_equal(parallel_tensor.local_tensors[2].tensor, inp[8:12, :])
    assert np.array_equal(parallel_tensor.local_tensors[3].tensor, inp[12:16, :])


def test_shard_replicate_two_dim_2():
    inp = np.random.rand(16, 16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (2, 1))

    assert parallel_tensor.device_mesh == (2, 1)
    assert len(parallel_tensor.local_tensors) == 4

    assert np.array_equal(parallel_tensor.local_tensors[0].tensor, inp[0:8, :])
    assert np.array_equal(parallel_tensor.local_tensors[1].tensor, inp[8:16, :])
    assert np.array_equal(parallel_tensor.local_tensors[2].tensor, inp[0:8, :])
    assert np.array_equal(parallel_tensor.local_tensors[3].tensor, inp[8:16, :])


def test_replicate_shard_two_dim_1():
    inp = np.random.rand(16, 16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (1, 4))

    assert parallel_tensor.device_mesh == (1, 4)
    assert len(parallel_tensor.local_tensors) == 4

    assert np.array_equal(parallel_tensor.local_tensors[0].tensor, inp[:, 0:4])
    assert np.array_equal(parallel_tensor.local_tensors[1].tensor, inp[:, 4:8])
    assert np.array_equal(parallel_tensor.local_tensors[2].tensor, inp[:, 8:12])
    assert np.array_equal(parallel_tensor.local_tensors[3].tensor, inp[:, 12:16])


def test_replicate_shard_two_dim_2():
    inp = np.random.rand(16, 16)
    tensor = Tensor(inp)
    parallel.N_DEVICE = 4
    parallel_tensor = distribute_tensor(tensor, (1, 2))

    assert parallel_tensor.device_mesh == (1, 2)
    assert len(parallel_tensor.local_tensors) == 4

    assert np.array_equal(parallel_tensor.local_tensors[0].tensor, inp[:, 0:8])
    assert np.array_equal(parallel_tensor.local_tensors[1].tensor, inp[:, 8:16])
    assert np.array_equal(parallel_tensor.local_tensors[2].tensor, inp[:, 0:8])
    assert np.array_equal(parallel_tensor.local_tensors[3].tensor, inp[:, 8:16])
