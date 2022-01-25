from typing import List, Optional, Type

import numpy as np
import torch as T
from gym import Env

from pearll.agents.base_agents import BaseAgent
from pearll.buffers.base_buffer import BaseBuffer
from pearll.buffers.replay_buffer import ReplayBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.type_aliases import Log
from pearll.common.utils import get_space_shape, to_numpy
from pearll.explorers.base_explorer import BaseExplorer
from pearll.models.actor_critics import ActorCritic, Critic, EpsilonGreedyActor
from pearll.models.encoders import IdentityEncoder
from pearll.models.heads import DiscreteQHead
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
from pearll.updaters.critics import BaseCriticUpdater, DiscreteQRegression


def get_default_model(env: Env):
    observation_shape = get_space_shape(env.observation_space)
    action_size = env.action_space.n
    observation_size = get_mlp_size(observation_shape)

    encoder = IdentityEncoder()
    torso = MLP(layer_sizes=[observation_size, 64, 32], activation_fn=T.nn.ReLU)
    head = DiscreteQHead(input_shape=32, output_shape=action_size)

    actor = EpsilonGreedyActor(
        critic_encoder=encoder, critic_torso=torso, critic_head=head
    )
    critic = Critic(encoder=encoder, torso=torso, head=head, create_target=True)

    return ActorCritic(actor, critic)


class DQN(BaseAgent):
    """
    DQN Algorithm

    :param env: the gym-like environment to be used
    :param model: the neural network model
    :param td_gamma: trajectory discount factor
    :param updater_class: critic updater class
    :param optimizer_settings: optimizer settings
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
        model: ActorCritic,
        td_gamma: float = 0.99,
        updater_class: Type[BaseCriticUpdater] = DiscreteQRegression,
        optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = ReplayBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(start_steps=1000),
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
        self.updater = updater_class(
            loss_class=optimizer_settings.loss_class,
            optimizer_class=optimizer_settings.optimizer_class,
            max_grad=optimizer_settings.max_grad,
        )

        self.learning_rate = optimizer_settings.learning_rate
        self.td_gamma = td_gamma

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        critic_losses = np.zeros(shape=(critic_epochs))
        for i in range(critic_epochs):
            trajectories = self.buffer.sample(batch_size=batch_size, flatten_env=False)

            with T.no_grad():
                next_q_values = self.model.forward_target_critics(
                    trajectories.next_observations
                )
                next_q_values = to_numpy(next_q_values.max(dim=-1)[0])
                next_q_values = next_q_values[..., np.newaxis]
                target_q_values = TD_zero(
                    trajectories.rewards,
                    next_q_values,
                    trajectories.dones,
                    self.td_gamma,
                )

            updater_log = self.updater(
                self.model,
                trajectories.observations,
                target_q_values,
                trajectories.actions,
                learning_rate=self.learning_rate,
            )
            critic_losses[i] = updater_log.loss

        self.model.assign_targets()

        return Log(critic_loss=np.mean(critic_losses))
