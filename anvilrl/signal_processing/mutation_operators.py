"""
Methods for mutating a population
https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
"""

import numpy as np
from gym import Space


def _sample_indices(
    population: np.ndarray,
    mutation_rate: float = 0.05,
) -> np.ndarray:
    """
    Samples mutation indices from a population.

    :param population: the population of individuals to mutate
    :param mutation_rate: the probability of mutating an individual
    :return: the population indices to mutate
    """
    # Get indices of individuals to mutate
    mutation_indices = np.random.choice(
        np.arange(population.shape[0]),
        size=int(population.shape[0] * mutation_rate),
        replace=False,
    )

    return mutation_indices


def gaussian_mutation(
    population: np.ndarray,
    action_space: Space,
    mutation_rate: float = 0.05,
    mutation_std: float = 0.5,
) -> np.ndarray:
    """
    Mutates a population using a Gaussian mutation operator.

    :param population: the population of individuals to mutate
    :param action_space: the action space of the environment
    :param mutation_rate: the probability of mutating an individual
    :param mutation_std: the standard deviation of the Gaussian distribution used to mutate an individual
    :return: the mutated population
    """
    # Get indices of individuals to mutate
    mutation_indices = _sample_indices(population, mutation_rate)

    # Mutate individuals
    new_population = population.copy()
    for i in mutation_indices:
        new_population[i] += np.random.normal(0, mutation_std, population.shape[1])

    return np.clip(new_population, action_space.low, action_space.high)


def uniform_mutation(
    population: np.ndarray,
    action_space: Space,
    mutation_rate: float = 0.05,
) -> np.ndarray:
    """
    Mutates a population using a uniform mutation operator.

    :param population: the population of individuals to mutate
    :param action_space: the action space of the environment
    :param mutation_rate: the probability of mutating an individual
    :return: the mutated population
    """
    # Get indices of individuals to mutate
    mutation_indices = _sample_indices(population, mutation_rate)

    # Mutate individuals
    new_population = population.copy()
    for i in mutation_indices:
        new_population[i] += np.random.uniform(-1, 1, population.shape[1])

    return np.clip(new_population, action_space.low, action_space.high)


def discrete_mutation(
    population: np.ndarray,
    action_space: Space,
    mutation_rate: float = 0.05,
) -> np.ndarray:
    """
    Mutates a population using a discrete mutation operator.

    :param population: the population of individuals to mutate
    :param action_space: the action space of the environment
    :param mutation_rate: the probability of mutating an individual
    :return: the mutated population
    """
    # Get indices of individuals to mutate
    mutation_indices = _sample_indices(population, mutation_rate)

    # Mutate individuals
    new_population = population.copy()
    for i in mutation_indices:
        new_population[i] += np.random.randint(-1, 1, population.shape[1])

    return np.clip(new_population, 0, action_space.n)
