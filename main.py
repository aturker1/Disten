import numpy as np
from src.parallel import distribute_tensor

from src.tensor import Tensor


inp = np.random.rand(4, 4)

tensor = Tensor(inp)
parallel_tensor = distribute_tensor(tensor, (2, 2))
