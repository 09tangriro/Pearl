from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import torch as T


class TrajectoryType(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


@dataclass
class Trajectories:
    observations: Union[np.ndarray, T.Tensor]
    actions: Union[np.ndarray, T.Tensor]
    rewards: Union[np.ndarray, T.Tensor]
    next_observations: Union[np.ndarray, T.Tensor]
    dones: Union[np.ndarray, T.Tensor]
