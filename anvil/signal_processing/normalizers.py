import numpy as np
from gym import Space
from numba import jit


@jit(nopython=True, parallel=True)
def _scale_normalizer(
    data: np.ndarray, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    if np.all(np.abs(low) == np.abs(high)):
        return data / high
    else:
        return (2 * ((data - np.min(data)) / (np.max(data) - np.min(data)))) - 1


@jit(nopython=True, parallel=True)
def _mean_std_normalizer(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (data - mean) / std


def scale_normalizer(data: np.ndarray, space: Space) -> np.ndarray:
    """Normalize data in the range [-1, 1]"""
    return _scale_normalizer(data, space.high, space.low)


def mean_std_normalizer(data: np.ndarray) -> np.ndarray:
    """Standardize data with zero mean and unit std"""
    return _mean_std_normalizer(data, np.mean(data, axis=0), np.std(data, axis=0))
