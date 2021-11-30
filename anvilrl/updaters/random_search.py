import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from gym.vector import VectorEnv
from sklearn.preprocessing import scale
from torch.distributions import Normal, kl_divergence

from anvilrl.common.enumerations import PopulationInitStrategy
from anvilrl.common.type_aliases import UpdaterLog
from anvilrl.common.utils import get_space_range, get_space_shape, numpy_to_torch
from anvilrl.signal_processing import (
    crossover_operators,
    mutation_operators,
    selection_operators,
)

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
        self.action_space_shape = get_space_shape(env.single_action_space)
        self.action_space_range = get_space_range(env.single_action_space)

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
            self.population_size, *self.action_space_shape
        )
        population = self.mean + (population_std * self.normal_dist)
        if isinstance(self.env.single_action_space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, self.action_space_range[0], self.action_space_range[1]
        )
        return self.population

    def __call__(self, rewards: np.ndarray, lr: float) -> UpdaterLog:
        """
        Perform an optimization step

        :param rewards: the rewards for the current population
        :param lr: the learning rate
        :return: the updater log
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
            self.population_size, *self.action_space_shape
        )
        population = self.mean + (self.population_std * self.normal_dist)

        # Discretize and clip population as needed
        if isinstance(self.env.single_action_space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, self.action_space_range[0], self.action_space_range[1]
        )

        # Calculate Log metrics
        new_dist = Normal(numpy_to_torch(self.mean), std)
        population_entropy = new_dist.entropy().mean()
        population_kl = kl_divergence(old_dist, new_dist).mean()

        return UpdaterLog(divergence=population_kl, entropy=population_entropy)


class GeneticUpdater(BaseSearchUpdater):
    """
    Updater for the Genetic Algorithm

    :param env: the vector environment
    """

    def __init__(
        self,
        env: VectorEnv,
    ) -> None:
        super().__init__(env)
        self.population_std = None

    def initialize_population(
        self,
        population_init_strategy: PopulationInitStrategy = PopulationInitStrategy.UNIFORM,
        population_std: Union[float, np.ndarray] = 1,
        starting_point: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if population_init_strategy == PopulationInitStrategy.UNIFORM:
            population = np.random.uniform(
                self.action_space_range[0],
                self.action_space_range[1],
                (self.population_size, *self.action_space_shape),
            )
        elif population_init_strategy == PopulationInitStrategy.NORMAL:
            mean = (
                starting_point
                if starting_point is not None
                else self.env.single_action_space.sample()
            ).astype(np.float32)
            population = np.random.normal(
                mean, population_std, (self.population_size, *self.action_space_shape)
            )
        else:
            raise ValueError(
                f"The population initialization strategy {population_init_strategy} is not supported"
            )

        # Discretize and clip population as needed
        if isinstance(self.env.single_action_space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, self.action_space_range[0], self.action_space_range[1]
        )

        return self.population

    def __call__(
        self,
        rewards: np.ndarray,
        selection_operator: selection_operators,
        crossover_operator: crossover_operators,
        mutation_operator: mutation_operators,
        selection_settings: Dict[str, Any] = {},
        crossover_settings: Dict[str, Any] = {},
        mutation_settings: Dict[str, Any] = {},
        elitism: float = 0.1,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param rewards: the rewards for the current population
        :param selection_operator: the selection operator function
        :param crossover_operator: the crossover operator function
        :param mutation_operator: the mutation operator function
        :param selection_settings: the selection operator settings
        :param crossover_settings: the crossover operator settings
        :param mutation_settings: the mutation operator settings
        :param elitism: fraction of the population to keep as elite
        :return: the updater log
        """
        assert (
            self.population is not None
        ), "Before calling the updater you must call the population initializer `self.initialize_population()`"

        # Store elite population
        old_population = self.population.copy()
        num_elite = int(self.population_size * elitism)
        elite_indices = np.argpartition(rewards, -num_elite)[-num_elite:]
        elite_population = old_population[elite_indices]

        # Main update
        parents = selection_operator(self.population, rewards, **selection_settings)
        children = crossover_operator(parents, **crossover_settings)
        self.population = mutation_operator(
            children, self.env.single_action_space, **mutation_settings
        )
        self.population[elite_indices] = elite_population

        # Calculate Log metrics
        divergence = np.mean(np.abs(self.population - old_population))
        entropy = np.mean(
            np.abs(np.max(self.population, axis=0) - np.min(self.population, axis=0))
        )

        return UpdaterLog(divergence=divergence, entropy=entropy)
