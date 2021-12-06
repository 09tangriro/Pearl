from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch as T

Tensor = Union[np.ndarray, T.Tensor]
Observation = Union[np.ndarray, Dict[str, np.ndarray]]


@dataclass
class Trajectories:
    """Sample trajectory data needed for algorithms"""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    dones: Tensor


@dataclass
class DictTrajectories:
    """Sample trajectory data with dictionary observations needed for algorithms"""

    observations: Dict[str, Tensor]
    actions: Tensor
    rewards: Tensor
    next_observations: Dict[str, Tensor]
    dones: Tensor


@dataclass
class UpdaterLog:
    """Log to see updater metrics for algorithms"""

    loss: Optional[float] = None
    divergence: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class Log:
    """
    Log to see training progress

    :param reward: reward received
    :param actor_loss: actor network loss
    :param critic_loss: critic network loss
    :divergence: divergence of policy
    :entropy: entropy of policy
    """

    reward: float = 0
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    divergence: Optional[float] = None
    entropy: Optional[float] = None
