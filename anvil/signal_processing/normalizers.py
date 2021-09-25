import numpy as np
from gym import Space
from numba import jit


@jit(nopython=True, parallel=True)
def _even_normalizer(data: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    if np.all(np.abs(low) == np.abs(high)):
        return data / high


def even_normalizer(data: np.ndarray, space: Space):
    """Normalize data in the range [-1, 1]"""
    return _even_normalizer(data, space.high, space.low)
