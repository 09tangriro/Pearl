from typing import Optional, Type, Union

import numpy as np
import torch as T
from gym.vector.vector_env import VectorEnv

from anvilrl.agents.base_agents import BaseEvolutionAgent
from anvilrl.buffers import RolloutBuffer
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.type_aliases import Log
from anvilrl.common.utils import filter_dataclass_by_none
from anvilrl.models.actor_critics import DeepIndividual, Individual
from anvilrl.settings import (
    BufferSettings,
    CrossoverSettings,
    LoggerSettings,
    MutationSettings,
    PopulationInitializerSettings,
    SelectionSettings,
)
from anvilrl.signal_processing import (
    crossover_operators,
    mutation_operators,
    selection_operators,
)
from anvilrl.updaters.evolution import BaseEvolutionUpdater, GeneticUpdater


class GA(BaseEvolutionAgent):
    """
    Genetic Algorithm
    https://www.geeksforgeeks.org/genetic-algorithms/

    :param env: the gym vecotrized environment
    :param model: the model representing an individual in the population
    :param updater_class: the class to use for the updater handling the actual update algorithm
    :param population_settings: the settings object for population initialization
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param logger_settings: settings for the logger
    :param device: device to run on, accepts "auto", "cuda" or "cpu" (needed to pass to buffer,
        can mostly be ignored)
    :param learning_rate: learning rate for the updater
    :param seed: optional seed for the random number generator
    """

    def __init__(
        self,
        env: VectorEnv,
        model: Union[Individual, DeepIndividual],
        updater_class: Type[BaseEvolutionUpdater] = GeneticUpdater,
        selection_operator: selection_operators = selection_operators.roulette_selection,
        selection_settings: SelectionSettings = SelectionSettings(),
        crossover_operator: crossover_operators = crossover_operators.crossover_one_point,
        crossover_settings: CrossoverSettings = CrossoverSettings(),
        mutation_operator: mutation_operators = mutation_operators.uniform_mutation,
        mutation_settings: MutationSettings = MutationSettings(),
        elitism: float = 0.1,
        population_settings: PopulationInitializerSettings = PopulationInitializerSettings(),
        buffer_class: Type[BaseBuffer] = RolloutBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        logger_settings: LoggerSettings = LoggerSettings(),
        device: Union[str, T.device] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            updater_class=updater_class,
            population_settings=population_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            device=device,
            seed=seed,
        )

        self.selection_operator = selection_operator
        self.selection_settings = filter_dataclass_by_none(selection_settings)
        self.crossover_operator = crossover_operator
        self.crossover_settings = filter_dataclass_by_none(crossover_settings)
        self.mutation_operator = mutation_operator
        self.mutation_settings = filter_dataclass_by_none(mutation_settings)
        self.elitism = elitism

    def _fit(self, epochs: int = 1) -> Log:
        divergences = np.zeros(epochs)
        entropies = np.zeros(epochs)

        trajectories = self.buffer.all()
        rewards = trajectories.rewards.squeeze()
        if rewards.ndim > 1:
            rewards = rewards.sum(dim=0)
        for i in range(epochs):
            log = self.updater(
                rewards=rewards,
                selection_operator=self.selection_operator,
                crossover_operator=self.crossover_operator,
                mutation_operator=self.mutation_operator,
                selection_settings=self.selection_settings,
                crossover_settings=self.crossover_settings,
                mutation_settings=self.mutation_settings,
                elitism=self.elitism,
            )
            divergences[i] = log.divergence
            entropies[i] = log.entropy
        self.buffer.reset()

        return Log(divergence=divergences.sum(), entropy=entropies.mean())
