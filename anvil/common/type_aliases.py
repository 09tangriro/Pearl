from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

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


@dataclass
class ActorUpdaterLog:
    loss: T.Tensor
    kl: Optional[T.Tensor]
    entropy: Optional[T.Tensor]


@dataclass
class CriticUpdaterLog:
    loss: T.Tensor
