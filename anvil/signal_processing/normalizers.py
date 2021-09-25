import numpy as np
from gym import Space
from numba import jit


@jit(nopython=True, parallel=True)
def even_normalizer(data: np.ndarray, space: Space) -> np.ndarray:
    """Normalize data in the range [-1, 1]"""
    return 2 * ((data - space.low) / space.high - space.low) - 1
