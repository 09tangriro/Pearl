from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch as T


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
    kl: Optional[T.Tensor] = None
    entropy: Optional[T.Tensor] = None


@dataclass
class CriticUpdaterLog:
    loss: T.Tensor
