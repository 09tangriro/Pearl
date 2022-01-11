"""This module holds settings objects to configure the other modules"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as T
from torch.optim.optimizer import Optimizer

from pearll.common.enumerations import Distribution


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

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PopulationSettings:
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

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ExplorerSettings:
    """
    Settings for the action explorer

    :param start_steps: number of steps at the start to randomly sample actions (encourages exploration)
    :param scale: std of noise to add to actions (not always applicable)
    """

    start_steps: int = 1000
    scale: Optional[float] = None

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class BufferSettings:
    """
    Settings for buffers

    :buffer_size: max number of transitions to store at once in each environment
    """

    buffer_size: int = int(1e6)

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CallbackSettings:
    """Settings for callbacks. Extend this class to add __init__ params for each callback."""

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


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
    log_frequency: Tuple[str, int] = ("episode", 1)
    file_handler_level: int = logging.DEBUG
    stream_handler_level: int = logging.INFO
    verbose: bool = True

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SelectionSettings:
    """Settings for the selection process. Extend this class to add params for each selection method."""

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CrossoverSettings:
    """Settings for the crossover process. Extend this class to add params for each crossover method."""

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class MutationSettings:
    """Settings for the mutation process. Extend this class to add params for each mutation method."""

    mutation_rate: float = 0.1
    mutation_std: Optional[float] = None

    def filter_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
