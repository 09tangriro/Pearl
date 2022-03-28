import warnings
from functools import partial
from typing import Callable, List, Optional, Type

import numpy as np
from gym.vector.vector_env import VectorEnv
from sklearn.preprocessing import scale

from pearll.agents.base_agents import BaseAgent
from pearll.buffers import RolloutBuffer
from pearll.buffers.base_buffer import BaseBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.type_aliases import Log
from pearll.common.utils import filter_rewards
from pearll.explorers.base_explorer import BaseExplorer
from pearll.models import ActorCritic, Dummy
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    MutationSettings,
    PopulationSettings,
    Settings,
)
from pearll.updaters.evolution import BaseEvolutionUpdater, NoisyGradientAscent

warnings.filterwarnings("ignore", category=UserWarning)


def default_model(env: VectorEnv):
    """
    Returns a default model for the given environment.
    """
    actor = Dummy(space=env.single_action_space)
    critic = Dummy(space=env.single_action_space)

    return ActorCritic(
        actor=actor,
        critic=critic,
        population_settings=PopulationSettings(
            actor_population_size=env.num_envs, actor_distribution="normal"
        ),
    )


class AdamES(BaseAgent):
    """
    AdamES is a variant of ES that uses the Adam optimizer.

    :param env: the gym-like environment to be used, should be a VectorEnv
    :param model: the neural network model
    :param updater_class: the updater class to be used
    :param mutation_operator: the mutation operator to be used
    :param mutation_settings: the mutation settings to be used
    :param learning_rate: the learning rate to be used
    :param momentum_weight: the adam momentum weight to be used
    :param damping_weight: the adam damping weight to be used
    :param learning_rate: the learning rate
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param action_explorer_class: the explorer class for random search at beginning of training and
        adding noise to actions
    :param explorer settings: settings for the action explorer
    :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
    :param callback_settings: settings for callbacks
    :param logger_settings: settings for the logger
    :param misc_settings: settings for miscellaneous parameters
    """

    def __init__(
        self,
        env: VectorEnv,
        model: Optional[ActorCritic] = None,
        updater_class: Type[BaseEvolutionUpdater] = NoisyGradientAscent,
        mutation_operator: Optional[Callable] = None,
        mutation_settings: MutationSettings = MutationSettings(mutation_std=0.5),
        learning_rate: float = 1,
        momentum_weight: float = 0.9,
        damping_weight: float = 0.999,
        buffer_class: Type[BaseBuffer] = RolloutBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(start_steps=0),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[Settings]] = None,
        logger_settings: LoggerSettings = LoggerSettings(),
        misc_settings: MiscellaneousSettings = MiscellaneousSettings(),
    ) -> None:
        model = model if model is not None else default_model(env)
        super().__init__(
            env=env,
            model=model,
            action_explorer_class=action_explorer_class,
            explorer_settings=explorer_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            callbacks=callbacks,
            callback_settings=callback_settings,
            misc_settings=misc_settings,
        )

        self.learning_rate = learning_rate
        self.momentum_weight = momentum_weight
        self.damping_weight = damping_weight
        self.updater = updater_class(model=self.model)
        self.mutation_operator = (
            None
            if mutation_operator is None
            else partial(mutation_operator, **mutation_settings.filter_none())
        )
        self.m = 0
        self.v = 0
        self.adam_step = 1

    def _adam(self, grad: np.floating) -> np.floating:
        """Adam optimizer update"""
        self.m = (1 - self.momentum_weight) * grad + self.momentum_weight * self.m
        self.v = (1 - self.damping_weight) * (
            grad * grad
        ) + self.damping_weight * self.v
        m_adj = self.m / (1 - (self.momentum_weight ** self.adam_step))
        v_adj = self.v / (1 - (self.damping_weight ** self.adam_step))
        self.adam_step += 1
        return m_adj / (np.sqrt(v_adj) + 1e-8)

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        divergences = np.zeros(actor_epochs)
        entropies = np.zeros(actor_epochs)

        trajectories = self.buffer.all(flatten_env=False)
        rewards = trajectories.rewards.squeeze()
        rewards = filter_rewards(rewards, trajectories.dones.squeeze())
        if rewards.ndim > 1:
            rewards = rewards.sum(axis=-1)
        scaled_rewards = scale(rewards)
        grad_approx = np.dot(self.updater.normal_dist.T, scaled_rewards) / (
            np.mean(self.updater.std) * self.env.num_envs
        )
        optimization_direction = self._adam(grad_approx)
        for i in range(actor_epochs):
            log = self.updater(
                learning_rate=self.learning_rate,
                optimization_direction=optimization_direction,
                mutation_operator=self.mutation_operator,
            )
            divergences[i] = log.divergence
            entropies[i] = log.entropy
        self.buffer.reset()

        return Log(divergence=divergences.sum(), entropy=entropies.mean())
