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
    actor_loss: float = 0
    critic_loss: float = 0
    reward: float = 0
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class OptimizerSettings:
    optimizer_class: Type[Optimizer] = T.optim.Adam
    learning_rate: float = 1e-3
    max_grad: float = 0.5


@dataclass
class ExplorerSettings:
    start_steps: int = 1000
    scale: Optional[float] = None


@dataclass
class BufferSettings:
    buffer_size: int = int(1e6)
    n_envs: int = 1


@dataclass
class CallbackSettings:
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
