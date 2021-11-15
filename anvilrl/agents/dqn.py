from typing import List, Optional, Type, Union

import numpy as np
import torch as T
from gym import Env

from anvilrl.agents.base_agents import BaseDeepAgent
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.buffers.replay_buffer import ReplayBuffer
from anvilrl.callbacks.base_callback import BaseCallback
from anvilrl.common.type_aliases import Log
from anvilrl.common.utils import get_space_shape, torch_to_numpy
from anvilrl.explorers.base_explorer import BaseExplorer
from anvilrl.models.actor_critics import (
    ActorCritic,
    ActorCriticWithCriticTarget,
    Critic,
    EpsilonGreedyActor,
)
from anvilrl.models.encoders import IdentityEncoder
from anvilrl.models.heads import DiscreteQHead
from anvilrl.models.torsos import MLP
from anvilrl.models.utils import get_mlp_size
from anvilrl.settings import (
    BufferSettings,
    CallbackSettings,
    ExplorerSettings,
    LoggerSettings,
    OptimizerSettings,
)
from anvilrl.signal_processing.return_estimators import TD_zero
from anvilrl.updaters.critics import BaseCriticUpdater, QRegression


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
    critic = Critic(encoder=encoder, torso=torso, head=head)

    return ActorCriticWithCriticTarget(actor, critic)


class DQN(BaseDeepAgent):
    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        updater_class: Type[BaseCriticUpdater] = QRegression,
        optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = ReplayBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(start_steps=1000),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[CallbackSettings]] = None,
        logger_settings: LoggerSettings = LoggerSettings(),
        device: Union[T.device, str] = "auto",
        render: bool = False,
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
        )
        self.updater = updater_class(
            optimizer_class=optimizer_settings.optimizer_class,
            lr=optimizer_settings.learning_rate,
            max_grad=optimizer_settings.max_grad,
        )

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        critic_losses = np.zeros(shape=(critic_epochs))
        for i in range(critic_epochs):
            trajectories = self.buffer.sample(batch_size=batch_size)

            with T.no_grad():
                next_q_values = self.model.target_critic(trajectories.next_observations)
                next_q_values, _ = next_q_values.max(dim=-1)
                next_q_values = torch_to_numpy(next_q_values.reshape(-1, 1))
                target_q_values = TD_zero(
                    trajectories.rewards, next_q_values, trajectories.dones
                )

            updater_log = self.updater(
                self.model,
                trajectories.observations,
                target_q_values,
                actions_index=trajectories.actions,
            )
            critic_losses[i] = updater_log.loss

        self.model.assign_targets()

        return Log(critic_loss=np.mean(critic_losses))


if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v0")
    agent = DQN(
        env=env,
        model=None,
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/DQN-demo", verbose=True
        ),
        explorer_settings=ExplorerSettings(start_steps=1000),
    )
    agent.fit(
        num_steps=50000, batch_size=32, critic_epochs=16, train_frequency=("episode", 1)
    )
