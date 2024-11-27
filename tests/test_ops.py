import numpy as np
from src.tensor import Tensor
from src.parallel import distribute_tensor
from src import parallel

def test_add():
    parallel.N_DEVICE = 4
    inp1 = np.random.rand(16, 16)
    inp2 = np.random.rand(16, 16)
    tensor1 = Tensor(inp1)
    tensor2 = Tensor(inp2)
    parallel_tensor1 = distribute_tensor(tensor1, (2, 2))
    parallel_tensor2 = distribute_tensor(tensor2, (2, 2))

    parallel_tensor3 = parallel_tensor1 + parallel_tensor2

    assert np.array_equal(
        parallel_tensor3.local_tensors[0].tensor, inp1[0:8, 0:8] + inp2[0:8, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[1].tensor, inp1[0:8, 8:16] + inp2[0:8, 8:16]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[2].tensor, inp1[8:16, 0:8] + inp2[8:16, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[3].tensor, inp1[8:16, 8:16] + inp2[8:16, 8:16]
    )

def test_sub():
    parallel.N_DEVICE = 4
    inp1 = np.random.rand(16, 16)
    inp2 = np.random.rand(16, 16)
    tensor1 = Tensor(inp1)
    tensor2 = Tensor(inp2)
    parallel_tensor1 = distribute_tensor(tensor1, (2, 2))
    parallel_tensor2 = distribute_tensor(tensor2, (2, 2))

    parallel_tensor3 = parallel_tensor1 - parallel_tensor2

    assert np.array_equal(
        parallel_tensor3.local_tensors[0].tensor, inp1[0:8, 0:8] - inp2[0:8, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[1].tensor, inp1[0:8, 8:16] - inp2[0:8, 8:16]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[2].tensor, inp1[8:16, 0:8] - inp2[8:16, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[3].tensor, inp1[8:16, 8:16] - inp2[8:16, 8:16]
    )

def test_mul():
    parallel.N_DEVICE = 4
    inp1 = np.random.rand(16, 16)
    inp2 = np.random.rand(16, 16)
    tensor1 = Tensor(inp1)
    tensor2 = Tensor(inp2)
    parallel_tensor1 = distribute_tensor(tensor1, (2, 2))
    parallel_tensor2 = distribute_tensor(tensor2, (2, 2))

    parallel_tensor3 = parallel_tensor1 * parallel_tensor2

    assert np.array_equal(
        parallel_tensor3.local_tensors[0].tensor, inp1[0:8, 0:8] * inp2[0:8, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[1].tensor, inp1[0:8, 8:16] * inp2[0:8, 8:16]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[2].tensor, inp1[8:16, 0:8] * inp2[8:16, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[3].tensor, inp1[8:16, 8:16] * inp2[8:16, 8:16]
    )

def test_div():
    parallel.N_DEVICE = 4
    inp1 = np.random.rand(16, 16)
    inp2 = np.random.rand(16, 16)
    tensor1 = Tensor(inp1)
    tensor2 = Tensor(inp2)
    parallel_tensor1 = distribute_tensor(tensor1, (2, 2))
    parallel_tensor2 = distribute_tensor(tensor2, (2, 2))

    parallel_tensor3 = parallel_tensor1 / parallel_tensor2

    assert np.array_equal(
        parallel_tensor3.local_tensors[0].tensor, inp1[0:8, 0:8] / inp2[0:8, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[1].tensor, inp1[0:8, 8:16] / inp2[0:8, 8:16]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[2].tensor, inp1[8:16, 0:8] / inp2[8:16, 0:8]
    )
    assert np.array_equal(
        parallel_tensor3.local_tensors[3].tensor, inp1[8:16, 8:16] / inp2[8:16, 8:16]
    )