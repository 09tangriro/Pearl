import warnings
from typing import Optional, Type, Union

import numpy as np
import torch as T
from gym.vector.vector_env import VectorEnv
from sklearn.preprocessing import scale

from anvilrl.agents.base_agents import BaseEvolutionAgent
from anvilrl.buffers import RolloutBuffer
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.type_aliases import Log
from anvilrl.models.actor_critics import DeepIndividual, Individual
from anvilrl.settings import (
    BufferSettings,
    LoggerSettings,
    PopulationInitializerSettings,
)
from anvilrl.updaters.evolution import BaseEvolutionUpdater, NoisyGradientAscent

warnings.filterwarnings("ignore", category=UserWarning)


class ES(BaseEvolutionAgent):
    """
    Natural Evolutionary Strategy
    https://towardsdatascience.com/evolutionary-strategy-a-theoretical-implementation-guide-9176217e7ed8

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
        updater_class: Type[BaseEvolutionUpdater] = NoisyGradientAscent,
        learning_rate: float = 0.001,
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

        self.learning_rate = learning_rate

    def _fit(self, epochs: int = 1) -> Log:
        divergences = np.zeros(epochs)
        entropies = np.zeros(epochs)

        trajectories = self.buffer.all()
        rewards = trajectories.rewards.squeeze()
        if rewards.ndim > 1:
            rewards = rewards.sum(axis=0)
        scaled_rewards = scale(rewards)
        learning_rate = self.learning_rate / (
            np.mean(self.population_settings.population_std) * self.env.num_envs
        )
        optimization_direction = np.dot(self.updater.normal_dist.T, scaled_rewards)
        for i in range(epochs):
            log = self.updater(
                learning_rate=learning_rate,
                optimization_direction=optimization_direction,
            )
            divergences[i] = log.divergence
            entropies[i] = log.entropy
        self.buffer.reset()

        return Log(divergence=divergences.sum(), entropy=entropies.mean())
