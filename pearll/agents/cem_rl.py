import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Type

import numpy as np
import torch as T
from gym import Env
from gym.vector.vector_env import VectorEnv

from pearll.agents.base_agents import BaseAgent
from pearll.buffers import ReplayBuffer
from pearll.buffers.base_buffer import BaseBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.type_aliases import Log
from pearll.common.utils import filter_rewards, get_space_shape, to_numpy
from pearll.explorers import BaseExplorer
from pearll.models import Actor, ActorCritic, Critic
from pearll.models.encoders import IdentityEncoder, MLPEncoder
from pearll.models.heads import ContinuousQHead, DeterministicHead
from pearll.models.torsos import MLP
from pearll.models.utils import get_mlp_size
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    OptimizerSettings,
    PopulationSettings,
    Settings,
)
from pearll.signal_processing import (
    crossover_operators,
    return_estimators,
    selection_operators,
)
from pearll.updaters.critics import BaseCriticUpdater, ContinuousQRegression
from pearll.updaters.evolution import BaseEvolutionUpdater, GeneticUpdater


def get_default_model(env: Env) -> ActorCritic:
    action_shape = get_space_shape(env.single_action_space)
    observation_shape = get_space_shape(env.single_observation_space)
    action_size = get_mlp_size(action_shape)
    observation_size = get_mlp_size(observation_shape)
    encoder_actor = IdentityEncoder()
    encoder_critic = MLPEncoder(observation_size + action_size, observation_size)
    torso = MLP(layer_sizes=[observation_size, 40, 30], activation_fn=T.nn.ReLU)
    head_actor = DeterministicHead(
        input_shape=30, action_shape=action_shape, activation_fn=T.nn.Tanh
    )
    head_critic = ContinuousQHead(input_shape=30)
    return ActorCritic(
        actor=Actor(
            encoder=encoder_actor,
            torso=torso,
            head=head_actor,
            create_target=True,
        ),
        critic=Critic(
            encoder=encoder_critic,
            torso=torso,
            head=head_critic,
            create_target=True,
        ),
        population_settings=PopulationSettings(
            actor_population_size=env.num_envs,
            critic_population_size=env.num_envs,
            actor_distribution="normal",
            actor_std=2,
        ),
    )


@dataclass
class NaiveSelectionSettings(Settings):
    ratio: float = 0.5


class CEM_RL(BaseAgent):
    """
    CEM-RL Algorithm

    :param env: the gym-like environment to be used
    :param eval_env: the environment to be used for evaluating agent before evolutionary update
    :param model: the neural network model
    :param td_gamma: trajectory discount factor
    :param value_coeff: value loss weight
    :param actor_updater_class: actor updater class
    :param actor_optimizer_settings: actor optimizer settings
    :param critic_updater_class: critic updater class
    :param critic_optimizer_settings: critic optimizer settings
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param action_explorer_class: the explorer class for random search at beginning of training and
        adding noise to actions
    :param explorer_settings: settings for the action explorer
    :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
    :param callback_settings: settings for callbacks
    :param logger_settings: settings for the logger
    :param misc_settings: settings for miscellaneous parameters
    """

    def __init__(
        self,
        env: VectorEnv,
        eval_env: VectorEnv,
        model: Optional[ActorCritic],
        td_gamma: float = 0.99,
        value_coeff: float = 0.5,
        actor_updater_class: Type[BaseEvolutionUpdater] = GeneticUpdater,
        selection_operator: Callable = selection_operators.naive_selection,
        selection_settings: Settings = NaiveSelectionSettings(),
        crossover_operator: Callable = crossover_operators.fit_gaussian,
        critic_updater_class: Type[BaseCriticUpdater] = ContinuousQRegression,
        critic_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = ReplayBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(start_steps=0),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[Settings]] = None,
        logger_settings: LoggerSettings = LoggerSettings(),
        misc_settings: MiscellaneousSettings = MiscellaneousSettings(),
    ) -> None:
        model = model or get_default_model(env)
        super().__init__(
            env,
            model,
            action_explorer_class=action_explorer_class,
            explorer_settings=explorer_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            callbacks=callbacks,
            callback_settings=callback_settings,
            misc_settings=misc_settings,
        )
        self.eval_env = eval_env
        self.actor_updater = actor_updater_class(self.model)
        self.critic_updater = critic_updater_class(
            loss_class=critic_optimizer_settings.loss_class,
            optimizer_class=critic_optimizer_settings.optimizer_class,
            max_grad=critic_optimizer_settings.max_grad,
        )

        self.critic_optimizer_settings = critic_optimizer_settings
        self.value_coeff = value_coeff
        self.td_gamma = td_gamma
        self.selection_operator = partial(
            selection_operator, **selection_settings.filter_none()
        )
        self.crossover_operator = partial(
            crossover_operator, population_shape=self.model.numpy_actors().shape
        )

    def _fit(self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1):
        critic_losses = np.zeros(critic_epochs)
        divergences = np.zeros(actor_epochs)
        entropies = np.zeros(actor_epochs)

        model_copy = copy.deepcopy(self.model)

        # Train critic for critic_epochs
        for i in range(critic_epochs):
            trajectories = self.buffer.sample(batch_size=batch_size)
            with T.no_grad():
                next_actions = self.model.forward_target_actors(
                    trajectories.next_observations
                )
                next_q_values = self.model.forward_target_critics(
                    trajectories.next_observations, next_actions
                )
                target_q_values = return_estimators.TD_zero(
                    trajectories.rewards,
                    to_numpy(next_q_values),
                    trajectories.dones,
                    gamma=self.td_gamma,
                )
            critic_log = self.critic_updater(
                self.model,
                trajectories.observations,
                trajectories.actions,
                target_q_values,
                learning_rate=self.critic_optimizer_settings.learning_rate,
                loss_coeff=self.value_coeff,
            )
            critic_losses[i] = critic_log.loss

        # Reset half the actors to their state before being updated by the critic updater
        half_actors = self.model.num_actors // 2
        actors_no_update = model_copy.numpy_actors()[:half_actors]
        actors_with_update = self.model.numpy_actors()[half_actors:]
        new_state = np.concatenate([actors_no_update, actors_with_update], axis=0)
        self.model.set_actors_state(new_state)

        # Evaluate new model
        episode_dones = [False for _ in range(self.eval_env.num_envs)]
        observation = self.eval_env.reset()
        episode_length = 0
        while not np.all(episode_dones):
            action = to_numpy(self.model(observation))
            next_observation, reward, done, _ = self.eval_env.step(action)
            self.buffer.add_trajectory(
                observation, action, reward, next_observation, done
            )
            episode_length += 1
            observation = next_observation
            episode_dones = np.logical_or(episode_dones, done)

        trajectories = self.buffer.last(episode_length, flatten_env=False)
        rewards = trajectories.rewards.squeeze()
        rewards = filter_rewards(rewards, trajectories.dones.squeeze())
        if rewards.ndim > 1:
            rewards = rewards.sum(axis=-1)

        # Train actor for actor_epochs
        for i in range(actor_epochs):
            actor_log = self.actor_updater(
                rewards=rewards,
                selection_operator=self.selection_operator,
                crossover_operator=self.crossover_operator,
                elitism=0,
            )
            divergences[i] = actor_log.divergence
            entropies[i] = actor_log.entropy

        # Update target networks
        self.model.update_targets()

        return Log(
            critic_loss=np.mean(critic_losses),
            divergence=np.mean(divergences),
            entropy=np.mean(entropies),
        )
