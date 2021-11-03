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


@dataclass
class OptimizerSettings:
    """
    Settings for the model optimizers

    :param optimizer_class: class of optimizer algorithm to use
    :param learning_rate: optimizer learning rate
    :param max_grad: maximum gradient for gradient clipping
    """

    optimizer_class: Type[Optimizer] = T.optim.Adam
    learning_rate: float = 1e-3
    max_grad: float = 0.5


@dataclass
class ExplorerSettings:
    """
    Settings for the action explorer

    :param start_steps: number of steps at the start to randomly sample actions (encourages exploration)
    :param scale: std of noise to add to actions (not always applicable)
    """

    start_steps: int = 1000
    scale: Optional[float] = None


@dataclass
class BufferSettings:
    """
    Settings for buffers

    :buffer_size: max number of transitions to store at once in each environment
    :n_envs: number of environments being run
    """

    buffer_size: int = int(1e6)
    n_envs: int = 1


@dataclass
class CallbackSettings:
    """
    Settings for callbacks, pick which ones apply!

    :param save_freq: how often to save
    :param save_path: path to save to
    :name_prefix: prefix of the model file name
    """

    save_freq: Optional[int] = None
    save_path: Optional[str] = None
    name_prefix: Optional[str] = None


@dataclass
class LoggerSettings:
    """
    Settings for the Logger

    :param tensorboard_log_path: path to store the tensorboard log
    :param file_handler_level: logging level for the file log
    :param stream_handler_level: logging level for the streaming log
    :param verbose: whether to record any logs at all
    """

    tensorboard_log_path: Optional[str] = None
    file_handler_level: int = logging.DEBUG
    stream_handler_level: int = logging.INFO
    verbose: bool = True