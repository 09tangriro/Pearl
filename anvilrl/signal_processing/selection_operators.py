"""Methods for selecting individuals in a population to evolve for the next algorithm iteration"""

import numpy as np


def tournament_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    tournament_size: int = 2,
    probability: float = 0.8,
) -> np.ndarray:
    """
    Selects individuals for the next algorithm iteration using tournament selection.
    https://www.geeksforgeeks.org/tournament-selection-ga/

    :param population: the population of individuals to select from
    :param fitness_scores: the fitness scores of the individuals in the population
    :param tournament_size: the number of individuals to select from the population
    :param probability: the probability of selecting the best individual for each tournament
    :return: the selected individuals
    """
    # Need a combined data structure to keep track of the fitness scores of the individuals after sorting
    combined_data = {
        fitness_score: individual
        for individual, fitness_score in zip(population, fitness_scores)
    }

    # Sort the combined data structure to get sorted fitness scores and population
    sorted_fitness = np.flip(sorted(combined_data))
    population = np.array(
        [combined_data[fitness_score] for fitness_score in sorted_fitness]
    )

    # Calculate probabilities of selecting an individual for tournament
    probabilities = np.array(
        [probability * ((1 - probability) ** i) for i in range(fitness_scores.shape[0])]
    )
    # Ensure valid probability mass function (sums to 1)
    residual = 1 - np.sum(probabilities)
    probabilities[0] += residual

    # Create the tournament population
    tournament_indices = np.random.choice(
        population.shape[0],
        size=(population.shape[0], tournament_size),
        p=probabilities,
    )
    tournament_population = population[tournament_indices]

    return np.max(tournament_population, axis=1)


def roulette_selection(
    population: np.ndarray, fitness_scores: np.ndarray
) -> np.ndarray:
    """
    Selects individuals for the next algorithm iteration using roulette selection.

    :param population: the population of individuals to select from
    :param fitness_scores: the fitness scores of the individuals in the population
    :return: the selected individuals
    """
    # Calculate the total fitness score of the population
    total_fitness_score = np.sum(fitness_scores)

    # Calculate the probabilities of selecting an individual
    fitness_scores = fitness_scores / total_fitness_score

    # Select individuals based on the probabilities
    selected_indices = np.random.choice(
        population.shape[0], size=population.shape[0], p=fitness_scores
    )
    return population[selected_indices]