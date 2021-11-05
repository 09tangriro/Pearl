import logging
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
    """
    Log to see training progress

    :param actor_loss: actor network loss
    :param critic_loss: critic network loss
    :param reward: reward received
    :kl_divergence: KL divergence of policy
    :entropy: entropy of policy
    """

    actor_loss: float = 0
    critic_loss: float = 0
    reward: float = 0
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
