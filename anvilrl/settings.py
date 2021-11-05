import logging
from dataclasses import dataclass
from typing import Optional, Type

import torch as T
from torch.optim.optimizer import Optimizer


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
    """

    buffer_size: int = int(1e6)


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
