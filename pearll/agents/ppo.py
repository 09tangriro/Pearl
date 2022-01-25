from typing import List, Optional, Type

import gym
import numpy as np
import torch as T
from gym import Env

from pearll.agents.base_agents import BaseAgent
from pearll.buffers import BaseBuffer, RolloutBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.type_aliases import Log
from pearll.common.utils import get_space_shape
from pearll.explorers import BaseExplorer
from pearll.models import Actor, ActorCritic, Critic
from pearll.models.encoders import IdentityEncoder
from pearll.models.heads import CategoricalHead, ValueHead
from pearll.models.torsos import MLP
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    OptimizerSettings,
    Settings,
)
from pearll.signal_processing.advantage_estimators import generalized_advantage_estimate
from pearll.updaters.actors import BaseActorUpdater, ProximalPolicyClip
from pearll.updaters.critics import BaseCriticUpdater, ValueRegression


def get_default_model(env: Env) -> ActorCritic:
    """
    Returns a default model for the given environment.
    """
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    observation_shape = get_space_shape(env.observation_space)

    encoder = IdentityEncoder()
    torso = MLP(layer_sizes=[observation_shape[0], 64, 64], activation_fn=T.nn.Tanh)
    head_actor = CategoricalHead(
        input_shape=64, action_size=action_size, activation_fn=T.nn.Tanh
    )
    head_critic = ValueHead(input_shape=64)

    actor = Actor(encoder=encoder, torso=torso, head=head_actor)
    critic = Critic(encoder=encoder, torso=torso, head=head_critic)

    return ActorCritic(actor=actor, critic=critic)


class PPO(BaseAgent):
    """
    Proximal Policy Optimization Algorithm

    :param env: the gym-like environment to be used
    :param model: the neural network model
    :param entropy_coeff: entropy weight
    :param gae_lambda: GAE exponential average coeff
    :param gae_gamma: trajectory discount factor
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
        model: Optional[ActorCritic] = None,
        ratio_clip: float = 0.2,
        entropy_coeff: float = 0.01,
        gae_lambda: float = 0.95,
        gae_gamma: float = 0.99,
        value_coeff: float = 0.5,
        actor_updater_class: Type[BaseActorUpdater] = ProximalPolicyClip,
        actor_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        critic_updater_class: Type[BaseCriticUpdater] = ValueRegression,
        critic_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = RolloutBuffer,
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
        self.actor_updater = actor_updater_class(
            optimizer_class=actor_optimizer_settings.optimizer_class,
            max_grad=actor_optimizer_settings.max_grad,
        )
        self.critic_updater = critic_updater_class(
            optimizer_class=critic_optimizer_settings.optimizer_class,
            max_grad=critic_optimizer_settings.max_grad,
        )
        self.actor_optimizer_settings = actor_optimizer_settings
        self.critic_optimizer_settings = critic_optimizer_settings
        self.ratio_clip = ratio_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma

    def _fit(self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1):
        critic_losses = np.zeros(shape=(critic_epochs))
        actor_losses = np.zeros(shape=(actor_epochs))
        divergences = np.zeros(shape=(actor_epochs))
        entropies = np.zeros(shape=(actor_epochs))

        # Sample rollout and compute advantages/returns
        trajectories = self.buffer.sample(batch_size, dtype="torch")
        with T.no_grad():
            old_values = self.model.forward_critics(trajectories.observations)
            next_values = self.model.forward_critics(trajectories.next_observations)
            advantages, returns = generalized_advantage_estimate(
                rewards=trajectories.rewards,
                old_values=old_values,
                new_values=next_values,
                dones=trajectories.dones,
                gae_lambda=self.gae_lambda,
                gamma=self.gae_gamma,
            )
            old_distributions = self.model.action_distribution(
                trajectories.observations
            )
            old_log_probs = old_distributions.log_prob(trajectories.actions).sum(dim=-1)

        # Train actor for actor_epochs
        for i in range(actor_epochs):
            actor_log = self.actor_updater(
                model=self.model,
                observations=trajectories.observations,
                actions=trajectories.actions,
                advantages=advantages,
                old_log_probs=old_log_probs,
                learning_rate=self.actor_optimizer_settings.learning_rate,
                ratio_clip=self.ratio_clip,
                entropy_coeff=self.entropy_coeff,
            )
            actor_losses[i] = actor_log.loss
            divergences[i] = actor_log.divergence
            entropies[i] = actor_log.entropy

        # Train critic for critic_epochs
        for i in range(critic_epochs):
            critic_log = self.critic_updater(
                self.model,
                trajectories.observations,
                returns,
                learning_rate=self.critic_optimizer_settings.learning_rate,
                loss_coeff=self.value_coeff,
            )
            critic_losses[i] = critic_log.loss

        self.buffer.reset()

        return Log(
            actor_loss=actor_losses.mean(),
            critic_loss=critic_losses.mean(),
            divergence=divergences.sum(),
            entropy=entropies.mean(),
        )
