import warnings
from typing import Type, Union

import numpy as np
import torch as T
from gym.vector.vector_env import VectorEnv
from sklearn.preprocessing import scale

from anvilrl.agents.base_agents import BaseSearchAgent
from anvilrl.buffers import RolloutBuffer
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.type_aliases import Log
from anvilrl.settings import (
    BufferSettings,
    LoggerSettings,
    PopulationInitializerSettings,
)
from anvilrl.updaters.random_search import BaseSearchUpdater, EvolutionaryUpdater

warnings.filterwarnings("ignore", category=UserWarning)


class ES(BaseSearchAgent):
    """
    Natural Evolutionary Strategy
    https://towardsdatascience.com/evolutionary-strategy-a-theoretical-implementation-guide-9176217e7ed8

    :param env: the gym vecotrized environment
    :param updater_class: the class to use for the updater handling the actual update algorithm
    :param population_settings: the settings object for population initialization
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
        updater_class: Type[BaseSearchUpdater] = EvolutionaryUpdater,
        learning_rate: float = 0.001,
        population_settings: PopulationInitializerSettings = PopulationInitializerSettings(),
        buffer_class: BaseBuffer = RolloutBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        logger_settings: LoggerSettings = LoggerSettings(),
        device: Union[str, T.device] = "auto",
    ) -> None:
        super().__init__(
            env=env,
            updater_class=updater_class,
            population_settings=population_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            device=device,
        )

        self.learning_rate = learning_rate

    def _fit(self) -> Log:
        trajectories = self.buffer.all()
        scaled_rewards = scale(trajectories.rewards.squeeze())
        learning_rate = self.learning_rate / (
            np.mean(self.population_settings.population_std) * self.env.num_envs
        )
        optimization_direction = np.dot(self.updater.normal_dist.T, scaled_rewards)
        log = self.updater(
            learning_rate=learning_rate,
            optimization_direction=optimization_direction,
        )
        self.logger.info(f"POPULATION MEAN={self.updater.mean}")
        self.buffer.reset()

        return Log(divergence=log.divergence, entropy=log.entropy)
