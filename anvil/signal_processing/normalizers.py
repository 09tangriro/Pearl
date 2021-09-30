# TODO: NEEDS TO BE LOOKED AT AGAIN WITH TESTS!

from typing import Any, Optional

import numpy as np
import torch as T
from gym import Space

from anvil.common.type_aliases import Tensor


def _scale_normalizer_numpy(
    data: np.ndarray, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    if np.all(np.abs(low) == np.abs(high)):
        return data / high
    else:
        return (2 * ((data - np.min(data)) / (np.max(data) - np.min(data)))) - 1


def _scale_normalizer_torch(data: T.Tensor) -> T.Tensor:
    if T.all(T.abs(T.min(data)) == T.abs(T.max(data))):
        return data / T.max(data)
    else:
        return (2 * ((data - T.min(data)) / (T.max(data) - T.min(data)))) - 1


def _mean_std_normalizer_numpy(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (data - mean) / std


def scale_normalizer(data: Tensor, space: Space) -> Tensor:
    """Normalize data in the range [-1, 1]"""
    if isinstance(data, np.ndarray):
        return _scale_normalizer_numpy(data, space.high, space.low)
    else:
        return __scale_normalizer_torch(data)


def mean_std_normalizer(data: Tensor, space: Optional[Any] = None) -> Tensor:
    """Standardize data with zero mean and unit std"""
    if isinstance(data, np.ndarray):
        return _mean_std_normalizer_numpy(
            data, np.mean(data, axis=0), np.std(data, axis=0)
        )
