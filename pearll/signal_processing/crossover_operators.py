"""Methods for generating a new population from selected individuals"""

from typing import Optional, Tuple

import numpy as np


def fit_gaussian(parents: np.ndarray, population_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generates a new population given selected parents fitted to a Gaussian distribution.

    :param parents: the parent population
    :param population_shape: the shape of the new population
    :return: the new population
    """
    # Get the mean and standard deviation of the parents
    mean = np.mean(parents, axis=0)
    std = np.std(parents, axis=0)

    return np.random.normal(mean, std, size=population_shape)


def one_point_crossover(
    parents: np.ndarray,
    crossover_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a new population from two parent individuals using one-point crossover.
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm

    :param parents: the parent population
    :param crossover_index: crossover point index, if None then randomly selected
    :return: the new population
    """
    # Split population into parent pairs
    pairs = np.array([[a, b] for a, b in zip(parents[::2], parents[1::2])])

    # Get crossover indices
    if crossover_index is None:
        crossover_indices = np.random.choice(
            np.arange(parents.shape[1]),
            size=pairs.shape[0],
        )
    else:
        crossover_indices = np.full(pairs.shape[0], crossover_index)

    # Perform crossover
    for i, pair in enumerate(pairs):
        pairs[i] = [
            np.concatenate(
                (pair[0][: crossover_indices[i]], pair[1][crossover_indices[i] :]),
                axis=0,
            ),
            np.concatenate(
                (pair[1][: crossover_indices[i]], pair[0][crossover_indices[i] :]),
                axis=0,
            ),
        ]
    new_population = np.concatenate(pairs, axis=0)

    # Account for odd parent population size
    if parents.shape[0] % 2 != 0:
        new_population = np.concatenate((new_population, [parents[-1]]), axis=0)

    return new_population
