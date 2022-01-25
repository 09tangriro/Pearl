"""This module holds settings objects to configure the other modules"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as T
from torch.optim.optimizer import Optimizer

from pearll.common.enumerations import Distribution


@dataclass
class Settings:
    """Base class for settings objects"""

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class MiscellaneousSettings(Settings):
    """
    Miscellaneous settings for base agent

    :param seed: random seed
    :param device: device to use for computations
    :param render: whether to render the environment
    """

    device: str = "auto"
    render: bool = False
    seed: Optional[int] = None


@dataclass
class OptimizerSettings(Settings):
    """
    Settings for the model optimizers

    :param loss_class: optional surrogate loss class to use for the optimizer
    :param optimizer_class: class of optimizer algorithm to use
    :param learning_rate: optimizer learning rate
    :param max_grad: maximum gradient for gradient clipping
    """

    loss_class: Optional[T.nn.Module] = T.nn.MSELoss()
    optimizer_class: Type[Optimizer] = T.optim.Adam
    learning_rate: Optional[float] = 1e-3
    max_grad: float = 0.5


@dataclass
class PopulationSettings(Settings):
    """
    Settings for the population initializer

    :param actor_population_size: number of actors in the population
    :param critic_population_size: number of critics in the population
    :param actor_distribution: distribution of the actor population
    :param critic_distribution: distribution of the critic population
    :param actor_std: standard deviation of the actor population if normally distributed
    :param critic_std: standard deviation of the critic population if normally distributed
    """

    actor_population_size: int = 1
    critic_population_size: int = 1
    actor_distribution: Optional[Union[str, Distribution]] = None
    critic_distribution: Optional[Union[str, Distribution]] = None
    actor_std: Optional[Union[float, np.ndarray]] = 1
    critic_std: Optional[Union[float, np.ndarray]] = 1


@dataclass
class ExplorerSettings(Settings):
    """
    Settings for the action explorer

    :param start_steps: number of steps at the start to randomly sample actions (encourages exploration)
    :param scale: std of noise to add to actions (not always applicable)
    """

    start_steps: int = 1000
    scale: Optional[float] = None


@dataclass
class BufferSettings(Settings):
    """
    Settings for buffers

    :buffer_size: max number of transitions to store at once in each environment
    """

    buffer_size: int = int(1e6)


@dataclass
class LoggerSettings(Settings):
    """
    Settings for the Logger

    :param tensorboard_log_path: path to store the tensorboard log
    :param file_handler_level: logging level for the file log
    :param stream_handler_level: logging level for the streaming log
    :param verbose: whether to record any logs at all
    """

    tensorboard_log_path: Optional[str] = None
    log_frequency: Tuple[str, int] = ("episode", 1)
    file_handler_level: int = logging.DEBUG
    stream_handler_level: int = logging.INFO
    verbose: bool = True


@dataclass
class MutationSettings(Settings):
    """
    Settings for the mutation process. Extend this class to add params for each mutation method.

    :param mutation_rate: probability of mutation for each individual
    :param mutation_std: optional standard deviation of the mutation distribution
    """

    mutation_rate: float = 0.1
    mutation_std: Optional[float] = None
