from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch as T

Tensor = Union[np.ndarray, T.Tensor]


@dataclass
class Trajectories:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    dones: Tensor


@dataclass
class UpdaterLog:
    loss: T.Tensor
    kl: Optional[T.Tensor] = None
    entropy: Optional[T.Tensor] = None
