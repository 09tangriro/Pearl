from typing import List, Optional, Type, Union

import numpy as np
import torch as T
from gym import Env

from pearll.agents.base_agents import BaseAgent
from pearll.buffers import ReplayBuffer
from pearll.buffers.base_buffer import BaseBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.type_aliases import Log
from pearll.common.utils import get_space_shape, to_numpy
from pearll.explorers import BaseExplorer, GaussianExplorer
from pearll.models import Actor, ActorCritic, Critic
from pearll.models.encoders import IdentityEncoder
from pearll.models.heads import ContinuousQHead, DeterministicHead
from pearll.models.torsos import MLP
from pearll.models.utils import get_mlp_size
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    OptimizerSettings,
    Settings,
)
from pearll.signal_processing.return_estimators import TD_zero
from pearll.updaters.actors import BaseActorUpdater, DeterministicPolicyGradient
from pearll.updaters.critics import BaseCriticUpdater, ContinuousQRegression


def get_default_model(env: Env) -> ActorCritic:
    action_shape = get_space_shape(env.action_space)
    observation_shape = get_space_shape(env.observation_space)
    action_size = get_mlp_size(action_shape)
    observation_size = get_mlp_size(observation_shape)
    encoder_actor = IdentityEncoder()
    encoder_critic = IdentityEncoder()
    torso_actor = MLP(layer_sizes=[observation_size, 400, 300], activation_fn=T.nn.ReLU)
    torso_critic = MLP(
        layer_sizes=[observation_size + action_size, 400, 300], activation_fn=T.nn.ReLU
    )
    head_actor = DeterministicHead(
        input_shape=300, action_shape=action_shape, activation_fn=T.nn.Tanh
    )
    head_critic = ContinuousQHead(input_shape=300)
    return ActorCritic(
        actor=Actor(
            encoder=encoder_actor,
            torso=torso_actor,
            head=head_actor,
            create_target=True,
        ),
        critic=Critic(
            encoder=encoder_critic,
            torso=torso_critic,
            head=head_critic,
            create_target=True,
        ),
    )


class DDPG(BaseAgent):
    """
    DDPG Algorithm

    :param env: the gym-like environment to be used
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
        env: Env,
        model: Optional[ActorCritic],
        td_gamma: float = 0.99,
        value_coeff: float = 0.5,
        actor_updater_class: Type[BaseActorUpdater] = DeterministicPolicyGradient,
        actor_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        critic_updater_class: Type[BaseCriticUpdater] = ContinuousQRegression,
        critic_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = ReplayBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = GaussianExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(
            start_steps=1000, scale=0.1
        ),
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
        self.actor_updater = actor_updater_class(
            optimizer_class=actor_optimizer_settings.optimizer_class,
            max_grad=actor_optimizer_settings.max_grad,
        )
        self.critic_updater = critic_updater_class(
            loss_class=critic_optimizer_settings.loss_class,
            optimizer_class=critic_optimizer_settings.optimizer_class,
            max_grad=critic_optimizer_settings.max_grad,
        )

        self.actor_optimizer_settings = actor_optimizer_settings
        self.critic_optimizer_settings = critic_optimizer_settings
        self.value_coeff = value_coeff
        self.td_gamma = td_gamma

    def _fit(self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1):
        critic_losses = np.zeros(shape=(critic_epochs))
        actor_losses = np.zeros(shape=(actor_epochs))
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
                target_q_values = TD_zero(
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

        # Train actor for actor_epochs
        for i in range(actor_epochs):
            trajectories = self.buffer.sample(batch_size=batch_size)
            actor_log = self.actor_updater(self.model, trajectories.observations)
            actor_losses[i] = actor_log.loss

        # Update target networks
        self.model.update_targets()

        return Log(
            actor_loss=np.mean(actor_losses),
            critic_loss=np.mean(critic_losses),
        )
