from typing import Type, Union

import numpy as np
import torch as T
from gym.vector.vector_env import VectorEnv

from anvilrl.agents.base_agents import BaseSearchAgent
from anvilrl.buffers import RolloutBuffer
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.type_aliases import Log
from anvilrl.common.utils import filter_dataclass_by_none
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
from anvilrl.updaters.random_search import BaseSearchUpdater, GeneticUpdater


class GA(BaseSearchAgent):
    """
    Genetic Algorithm
    https://www.geeksforgeeks.org/genetic-algorithms/

    :param env: the gym vecotrized environment
    :param updater_class: the class to use for the updater handling the actual update algorithm
    :param population_initializer_settings: the settings object for population initialization
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param logger_settings: settings for the logger
    :param device: device to run on, accepts "auto", "cuda" or "cpu" (needed to pass to buffer,
        can mostly be ignored)
    :param learning_rate: learning rate for the updater
    """

    def __init__(
        self,
        env: VectorEnv,
        updater_class: Type[BaseSearchUpdater] = GeneticUpdater,
        selection_operator: selection_operators = selection_operators.roulette_selection,
        selection_settings: SelectionSettings = SelectionSettings(),
        crossover_operator: crossover_operators = crossover_operators.crossover_one_point,
        crossover_settings: CrossoverSettings = CrossoverSettings(),
        mutation_operator: mutation_operators = mutation_operators.uniform_mutation,
        mutation_settings: MutationSettings = MutationSettings(),
        elitism: float = 0.1,
        population_init_settings: PopulationInitializerSettings = PopulationInitializerSettings(),
        buffer_class: BaseBuffer = RolloutBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        logger_settings: LoggerSettings = LoggerSettings(),
        device: Union[str, T.device] = "auto",
    ) -> None:
        super().__init__(
            env=env,
            updater_class=updater_class,
            population_initializer_settings=population_init_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            device=device,
        )

        self.selection_operator = selection_operator
        self.selection_settings = filter_dataclass_by_none(selection_settings)
        self.crossover_operator = crossover_operator
        self.crossover_settings = filter_dataclass_by_none(crossover_settings)
        self.mutation_operator = mutation_operator
        self.mutation_settings = filter_dataclass_by_none(mutation_settings)
        self.elitism = elitism

    def _fit(self) -> Log:
        trajectories = self.buffer.all()
        log = self.updater(
            rewards=trajectories.rewards.flatten(),
            selection_operator=self.selection_operator,
            crossover_operator=self.crossover_operator,
            mutation_operator=self.mutation_operator,
            selection_settings=self.selection_settings,
            crossover_settings=self.crossover_settings,
            mutation_settings=self.mutation_settings,
            elitism=self.elitism,
        )
        self.logger.info(f"POPULATION MEAN={np.mean(self.updater.population, axis=0)}")
        self.buffer.reset()

        return Log(divergence=log.divergence, entropy=log.entropy)
