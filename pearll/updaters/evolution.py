from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from torch.distributions import Normal, kl_divergence

from pearll.common.type_aliases import (
    CrossoverFunc,
    MutationFunc,
    SelectionFunc,
    UpdaterLog,
)
from pearll.common.utils import to_torch
from pearll.models.actor_critics import ActorCritic


class BaseEvolutionUpdater(ABC):
    """
    The base random search updater class with pre-defined methods for derived classes

    :param model: the actor critic model containing the population
    :param population_type: the type of population to update, either "actor" or "critic"
    """

    def __init__(self, model: ActorCritic, population_type: str = "actor") -> None:
        self.model = model
        self.population_type = population_type
        if population_type == "actor":
            self.mean = model.mean_actor
            self.std = self.model.population_settings.actor_std
            self.population_size = model.num_actors
            self.normal_dist = model.normal_dist_actor
            self.space_shape = model.actor.space_shape
            self.space_range = model.actor.space_range
            self.space = model.actor.space
        elif population_type == "critic":
            self.mean = model.mean_critic
            self.std = self.model.population_settings.critic_std
            self.population_size = model.num_critics
            self.normal_dist = model.normal_dist_critic
            self.space_shape = model.critic.space_shape
            self.space_range = model.critic.space_range
            self.space = model.critic.space

    def update_networks(self, population: np.ndarray) -> None:
        """
        Update the networks in the population

        :param population: the population state to set the networks to
        """
        if self.population_type == "actor":
            self.model.set_actors_state(population)
        elif self.population_type == "critic":
            self.model.set_critics_state(population)

    @abstractmethod
    def __call__(self) -> UpdaterLog:
        """Run an optimization step"""


class NoisyGradientAscent(BaseEvolutionUpdater):
    """
    Updater for the Natural Evolutionary Strategy

    :param model: the actor critic model containing the population
    :param population_type: the type of population to update, either "actor" or "critic"
    """

    def __init__(self, model: ActorCritic, population_type: str = "actor") -> None:
        super().__init__(model, population_type)

    def __call__(
        self,
        learning_rate: float,
        optimization_direction: np.ndarray,
        mutation_operator: Optional[MutationFunc] = None,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param learning_rate: the learning rate
        :param optimization_direction: the optimization direction
        :param mutation_operator: the mutation operator
        :return: the updater log
        """
        # Snapshot current population dist for kl divergence
        # use copy() to avoid modifying the original
        old_dist = Normal(to_torch(self.mean.copy()), self.std)

        # Main update
        self.mean += learning_rate * optimization_direction

        # Generate new population
        self.normal_dist = np.random.randn(self.population_size, *self.space_shape)
        population = self.mean + (self.std * self.normal_dist)
        if mutation_operator is not None:
            population = mutation_operator(population, self.space)

        # Discretize and clip population as needed
        if isinstance(self.space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        population = np.clip(population, self.space_range[0], self.space_range[1])
        self.update_networks(population)

        # Calculate Log metrics
        new_dist = Normal(to_torch(self.mean), self.std)
        population_entropy = new_dist.entropy().mean()
        population_kl = kl_divergence(old_dist, new_dist).mean()

        return UpdaterLog(divergence=population_kl, entropy=population_entropy)


class GeneticUpdater(BaseEvolutionUpdater):
    """
    Updater for the Genetic Algorithm

    :param model: the actor critic model containing the population
    :param population_type: the type of population to update, either "actor" or "critic"
    """

    def __init__(self, model: ActorCritic, population_type: str = "actor") -> None:
        super().__init__(model, population_type)

    def __call__(
        self,
        rewards: np.ndarray,
        selection_operator: Optional[SelectionFunc] = None,
        crossover_operator: Optional[CrossoverFunc] = None,
        mutation_operator: Optional[MutationFunc] = None,
        elitism: float = 0.1,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param rewards: the rewards for the current population
        :param selection_operator: the selection operator function
        :param crossover_operator: the crossover operator function
        :param mutation_operator: the mutation operator function
        :param elitism: fraction of the population to keep as elite
        :return: the updater log
        """
        # Store elite population
        if self.population_type == "actor":
            old_population = self.model.numpy_actors()
        elif self.population_type == "critic":
            old_population = self.model.numpy_critics()
        if elitism > 0:
            num_elite = int(self.population_size * elitism)
            elite_indices = np.argpartition(rewards, -num_elite)[-num_elite:]
            elite_population = old_population[elite_indices]

        # Main update
        if selection_operator is not None:
            new_population = selection_operator(old_population, rewards)
        if crossover_operator is not None:
            new_population = crossover_operator(new_population)
        if mutation_operator is not None:
            new_population = mutation_operator(new_population, self.space)
        if elitism > 0:
            new_population[elite_indices] = elite_population
        self.update_networks(new_population)

        # Calculate Log metrics
        divergence = np.mean(np.abs(new_population - old_population))
        entropy = np.mean(
            np.abs(np.max(new_population, axis=0) - np.min(new_population, axis=0))
        )

        return UpdaterLog(divergence=divergence, entropy=entropy)
