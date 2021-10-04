from dataclasses import dataclass
from typing import Optional, Type, Union

import numpy as np
import torch as T
from torch.optim.optimizer import Optimizer

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
    loss: float
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class Log:
    reward: float
    actor_loss: float
    critic_loss: float
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class OptimizerSettings:
    optimizer_class: Type[Optimizer] = T.optim.Adam
    learning_rate: float = 1e-3
    max_grad: float = 0.5
