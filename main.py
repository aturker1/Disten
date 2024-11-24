import numpy as np
from src.parallel import distribute_tensor
from src.placements import Shard, Replicate

from src.tensor import Tensor, ParallelTensor


inp = np.random.rand(4,4)

tensor = Tensor(inp)
parallel_tensor = distribute_tensor(tensor, (2, 2))
