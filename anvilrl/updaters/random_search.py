import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from gym.vector import VectorEnv
from sklearn.preprocessing import scale
from torch.distributions import Normal, kl_divergence

from anvilrl.common.enumerations import PopulationInitStrategy
from anvilrl.common.type_aliases import UpdaterLog
from anvilrl.common.utils import numpy_to_torch

warnings.filterwarnings("ignore", category=UserWarning)


class BaseSearchUpdater(ABC):
    """
    The base random search updater class with pre-defined methods for derived classes

    :param env: the vector environment
    """

    def __init__(self, env: VectorEnv) -> None:
        self.env = env
        self.population = None
        self.population_size = env.num_envs

    @abstractmethod
    def initialize_population(
        self,
        population_init_strategy: PopulationInitStrategy,
        population_std: Optional[Union[float, np.ndarray]] = 1,
        starting_point: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize the population

        :param population_init_strategy: the population initialization strategy
        :param population_std: the standard deviation for the population initialization
        :param starting_point: the starting point for the population initialization
        :return: the starting population
        """

    @abstractmethod
    def __call__(self) -> UpdaterLog:
        """Run an optimization step"""


class EvolutionaryUpdater(BaseSearchUpdater):
    """
    Updater for the Natural Evolutionary Strategy

    :param env: the vector environment
    """

    def __init__(
        self,
        env: VectorEnv,
    ) -> None:
        super().__init__(env)
        self.normal_dist = None
        self.mean = None
        self.population_std = None

    def initialize_population(
        self,
        population_init_strategy: PopulationInitStrategy = PopulationInitStrategy.NORMAL,
        population_std: Union[float, np.ndarray] = 1,
        starting_point: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.population_std = population_std
        self.mean = (
            starting_point
            if starting_point is not None
            else self.env.single_action_space.sample()
        ).astype(np.float32)
        self.normal_dist = np.random.randn(
            self.population_size, self.env.single_action_space.shape[0]
        )
        self.population = self.mean + (population_std * self.normal_dist)
        return self.mean + (population_std * self.normal_dist)

    def __call__(self, rewards: np.ndarray, lr: float) -> UpdaterLog:
        """
        Perform an optimization step

        :param rewards: the rewards for the current population
        :param lr: the learning rate
        """
        assert (
            self.mean is not None
        ), "Before calling the updater you must call the population initializer `self.initialize_population()`"
        std = (
            numpy_to_torch(self.population_std)
            if isinstance(self.population_std, np.ndarray)
            else self.population_std
        )
        # Snapshot current population dist for kl divergence
        # use copy() to avoid modifying the original
        old_dist = Normal(numpy_to_torch(self.mean.copy()), std)
        scaled_rewards = scale(rewards.squeeze())

        # Main update
        self.mean += (
            lr / (np.mean(self.population_std) * self.population_size)
        ) * np.dot(self.normal_dist.T, scaled_rewards)

        # Generate new population
        self.normal_dist = np.random.randn(
            self.population_size, self.env.single_action_space.shape[0]
        )
        population = self.mean + (self.population_std * self.normal_dist)
        self.population = np.clip(
            population,
            self.env.single_action_space.low,
            self.env.single_action_space.high,
        )

        # Calculate Log metrics
        new_dist = Normal(numpy_to_torch(self.mean), std)
        population_entropy = new_dist.entropy().mean()
        population_kl = kl_divergence(old_dist, new_dist).mean()

        return UpdaterLog(kl_divergence=population_kl, entropy=population_entropy)
