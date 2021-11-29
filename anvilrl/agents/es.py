from typing import Type, Union

import numpy as np
import torch as T
from gym.vector.vector_env import VectorEnv

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


class ES(BaseSearchAgent):
    """
    Natural Evolutionary Strategy
    https://towardsdatascience.com/evolutionary-strategy-a-theoretical-implementation-guide-9176217e7ed8

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
        updater_class: Type[BaseSearchUpdater] = EvolutionaryUpdater,
        learning_rate: float = 0.001,
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

        self.learning_rate = learning_rate

    def _fit(self) -> Log:
        trajectories = self.buffer.all()
        log = self.updater(rewards=trajectories.rewards, lr=self.learning_rate)
        self.logger.info(f"POPULATION MEAN={self.updater.mean}")
        self.buffer.reset()

        return Log(divergence=log.divergence, entropy=log.entropy)


if __name__ == "__main__":
    import gym

    class Sphere(gym.Env):
        """
        Sphere(2) function for testing ES agent.
        """

        def __init__(self):
            self.action_space = gym.spaces.Box(low=-100, high=100, shape=(2,))
            self.observation_space = gym.spaces.Discrete(1)

        def step(self, action):
            return 0, -(action[0] ** 2 + action[1] ** 2), False, {}

        def reset(self):
            return 0

    POPULATION_SIZE = 10
    env = gym.vector.SyncVectorEnv([lambda: Sphere() for _ in range(POPULATION_SIZE)])

    agent = ES(
        env=env,
        population_init_settings=PopulationInitializerSettings(
            starting_point=np.array([10, 10])
        ),
        learning_rate=1,
        logger_settings=LoggerSettings(tensorboard_log_path="runs/ES-demo"),
    )
    agent.fit(num_steps=15)
