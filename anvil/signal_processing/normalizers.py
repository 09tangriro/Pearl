import numpy as np
from gym import Space
from numba import jit


@jit(nopython=True, parallel=True)
def _even_normalizer(data: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Normalize data in the range [-1, 1]"""
    if np.all(np.abs(low) == np.abs(high)):
        return data / high


def even_normalizer(data: np.ndarray, space: Space):
    return _even_normalizer(data, space.high, space.low)
