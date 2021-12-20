from typing import List, Optional, Type, Union

import gym
import numpy as np
import torch as T
from gym import Env

from anvilrl.agents.base_agents import BaseDeepAgent
from anvilrl.buffers import BaseBuffer, RolloutBuffer
from anvilrl.callbacks.base_callback import BaseCallback
from anvilrl.common.type_aliases import Log
from anvilrl.common.utils import get_space_shape
from anvilrl.explorers import BaseExplorer
from anvilrl.models.actor_critics import Actor, ActorCritic, Critic
from anvilrl.models.encoders import IdentityEncoder
from anvilrl.models.heads import CategoricalHead, ValueHead
from anvilrl.models.torsos import MLP
from anvilrl.settings import (
    BufferSettings,
    CallbackSettings,
    ExplorerSettings,
    LoggerSettings,
    OptimizerSettings,
)
from anvilrl.signal_processing.advantage_estimators import (
    generalized_advantage_estimate,
)
from anvilrl.updaters.actors import BaseActorUpdater, PolicyGradient
from anvilrl.updaters.critics import BaseCriticUpdater, ValueRegression


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


class A2C(BaseDeepAgent):
    """
    Actor-Critic Algorithm

    :param env: the gym-like environment to be used
    :param model: the neural network model
    :param entropy_coefficient: entropy weight
    :param gae_lambda: GAE exponential average coefficient
    :param gae_gamma: trajectory discount factor
    :param value_coefficient: value loss weight
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
    :param device: device to run on, accepts "auto", "cuda" or "cpu"
    :param render: whether to render the environment or not
    :param seed: optional seed for the random number generator
    """

    def __init__(
        self,
        env: Env,
        model: Optional[ActorCritic] = None,
        entropy_coefficient: float = 0.01,
        gae_lambda: float = 0.95,
        gae_gamma: float = 0.99,
        value_coefficient: float = 0.5,
        actor_updater_class: Type[BaseActorUpdater] = PolicyGradient,
        actor_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        critic_updater_class: Type[BaseCriticUpdater] = ValueRegression,
        critic_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = RolloutBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(start_steps=0),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[CallbackSettings]] = None,
        logger_settings: LoggerSettings = LoggerSettings(),
        device: Union[T.device, str] = "auto",
        render: bool = False,
        seed: Optional[int] = None,
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
            device=device,
            render=render,
            seed=seed,
        )
        self.actor_updater = actor_updater_class(
            optimizer_class=actor_optimizer_settings.optimizer_class,
            lr=actor_optimizer_settings.learning_rate,
            max_grad=actor_optimizer_settings.max_grad,
            entropy_coeff=entropy_coefficient,
        )
        self.critic_updater = critic_updater_class(
            optimizer_class=critic_optimizer_settings.optimizer_class,
            lr=critic_optimizer_settings.learning_rate,
            loss_coeff=value_coefficient,
            max_grad=critic_optimizer_settings.max_grad,
        )
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
            old_values = self.model.forward_critic(trajectories.observations)
            next_values = self.model.forward_critic(trajectories.next_observations)
            advantages, returns = generalized_advantage_estimate(
                rewards=trajectories.rewards,
                old_values=old_values,
                new_values=next_values,
                dones=trajectories.dones,
                gae_lambda=self.gae_lambda,
                gamma=self.gae_gamma,
            )

        # Train actor for actor_epochs
        for i in range(actor_epochs):
            actor_log = self.actor_updater(
                model=self.model,
                observations=trajectories.observations,
                actions=trajectories.actions,
                advantages=advantages,
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
            )
            critic_losses[i] = critic_log.loss

        self.buffer.reset()

        return Log(
            actor_loss=actor_losses.mean(),
            critic_loss=critic_losses.mean(),
            divergence=divergences.sum(),
            entropy=entropies.mean(),
        )
