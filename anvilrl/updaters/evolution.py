from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from gym.vector import VectorEnv
from torch.distributions import Normal, kl_divergence

from anvilrl.common.enumerations import PopulationInitStrategy
from anvilrl.common.type_aliases import UpdaterLog
from anvilrl.common.utils import numpy_to_torch
from anvilrl.models.actor_critics import DeepIndividual, Individual
from anvilrl.signal_processing import (
    crossover_operators,
    mutation_operators,
    selection_operators,
)


class BaseEvolutionUpdater(ABC):
    """
    The base random search updater class with pre-defined methods for derived classes

    :param env: the vector environment
    :param model: the model representing an individual in the population
    """

    def __init__(
        self, env: VectorEnv, model: Union[Individual, DeepIndividual]
    ) -> None:
        self.model = model
        self.population = None
        self.population_size = env.num_envs

    @abstractmethod
    def initialize_population(
        self,
        population_init_strategy: PopulationInitStrategy,
        population_std: Optional[Union[float, np.ndarray]] = 1,
        starting_point: Optional[np.ndarray] = None,
    ) -> List[Union[Individual, DeepIndividual]]:
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


class NoisyGradientAscent(BaseEvolutionUpdater):
    """
    Updater for the Natural Evolutionary Strategy

    :param env: the vector environment
    :param model: the model representing an individual in the population
    """

    def __init__(
        self,
        env: VectorEnv,
        model: Union[Individual, DeepIndividual],
    ) -> None:
        super().__init__(env, model)
        self.normal_dist = None
        self.mean = None
        self.population_std = None

    def initialize_population(
        self,
        population_init_strategy: PopulationInitStrategy = PopulationInitStrategy.NORMAL,
        population_std: Union[float, np.ndarray] = 1,
        starting_point: Optional[np.ndarray] = None,
    ) -> List[Union[Individual, DeepIndividual]]:
        self.population_std = population_std
        self.mean = starting_point.astype(np.float32)
        self.normal_dist = np.random.randn(
            self.population_size, *self.model.space_shape
        )
        population = self.mean + (population_std * self.normal_dist)

        if isinstance(self.model.space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, self.model.space_range[0], self.model.space_range[1]
        )

        return [deepcopy(self.model).set_state(ind) for ind in self.population]

    def __call__(
        self, learning_rate: float, optimization_direction: np.ndarray
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param rewards: the rewards for the current population
        :param learning_rate: the learning rate
        :param optimization_direction: the optimization direction
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

        # Main update
        self.mean += learning_rate * optimization_direction

        # Generate new population
        self.normal_dist = np.random.randn(
            self.population_size, *self.model.space_shape
        )
        population = self.mean + (self.population_std * self.normal_dist)

        # Discretize and clip population as needed
        if isinstance(self.model.space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, self.model.space_range[0], self.model.space_range[1]
        )

        # Calculate Log metrics
        new_dist = Normal(numpy_to_torch(self.mean), std)
        population_entropy = new_dist.entropy().mean()
        population_kl = kl_divergence(old_dist, new_dist).mean()

        return UpdaterLog(divergence=population_kl, entropy=population_entropy)


class GeneticUpdater(BaseEvolutionUpdater):
    """
    Updater for the Genetic Algorithm

    :param env: the vector environment
    :param model: the model representing an individual in the population
    """

    def __init__(
        self,
        env: VectorEnv,
        model: Union[Individual, DeepIndividual],
    ) -> None:
        super().__init__(env, model)
        self.population_std = None

    def initialize_population(
        self,
        population_init_strategy: PopulationInitStrategy = PopulationInitStrategy.UNIFORM,
        population_std: Union[float, np.ndarray] = 1,
        starting_point: Optional[np.ndarray] = None,
    ) -> List[Union[Individual, DeepIndividual]]:
        if population_init_strategy == PopulationInitStrategy.UNIFORM:
            population = np.random.uniform(
                self.model.space_range[0],
                self.model.space_range[1],
                (self.population_size, *self.model.space_shape),
            )
        elif population_init_strategy == PopulationInitStrategy.NORMAL:
            mean = (starting_point).astype(np.float32)
            population = np.random.normal(
                mean, population_std, (self.population_size, *self.model.space_shape)
            )
        else:
            raise ValueError(
                f"The population initialization strategy {population_init_strategy} is not supported"
            )

        # Discretize and clip population as needed
        if isinstance(self.model.space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, self.model.space_range[0], self.model.space_range[1]
        )

        return [deepcopy(self.model).set_state(ind) for ind in self.population]

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
            children, self.model.space, **mutation_settings
        )
        self.population[elite_indices] = elite_population

        # Calculate Log metrics
        divergence = np.mean(np.abs(self.population - old_population))
        entropy = np.mean(
            np.abs(np.max(self.population, axis=0) - np.min(self.population, axis=0))
        )

        return UpdaterLog(divergence=divergence, entropy=entropy)
