from typing import List, Optional, Type, Union

import numpy as np
import torch as T
from gym import Env

from anvil_rl.agents import BaseAgent
from anvil_rl.buffers import ReplayBuffer
from anvil_rl.buffers.base_buffer import BaseBuffer
from anvil_rl.callbacks.base_callback import BaseCallback
from anvil_rl.common.type_aliases import (
    BufferSettings,
    ExplorerSettings,
    Log,
    OptimizerSettings,
)
from anvil_rl.common.utils import get_space_shape, torch_to_numpy
from anvil_rl.explorers import BaseExplorer, GaussianExplorer
from anvil_rl.models.actor_critics import (
    Actor,
    ActorCritic,
    ActorCriticWithTargets,
    Critic,
)
from anvil_rl.models.encoders import IdentityEncoder
from anvil_rl.models.heads import ContinuousQHead, DeterministicPolicyHead
from anvil_rl.models.torsos import MLP
from anvil_rl.models.utils import get_mlp_size
from anvil_rl.signal_processing.sample_estimators import TD_zero
from anvil_rl.updaters.actors import BaseActorUpdater, DeterministicPolicyGradient
from anvil_rl.updaters.critics import BaseCriticUpdater, QRegression


def get_default_model(env: Env) -> ActorCriticWithTargets:
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
    head_actor = DeterministicPolicyHead(
        input_shape=300, action_shape=action_shape, activation_fn=T.nn.Tanh
    )
    head_critic = ContinuousQHead(input_shape=300)
    return ActorCriticWithTargets(
        actor=Actor(encoder=encoder_actor, torso=torso_actor, head=head_actor),
        critic=Critic(encoder=encoder_critic, torso=torso_critic, head=head_critic),
    )


class DDPG(BaseAgent):
    def __init__(
        self,
        env: Env,
        model: Optional[ActorCritic],
        actor_updater_class: Type[BaseActorUpdater] = DeterministicPolicyGradient,
        actor_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        critic_updater_class: Type[BaseCriticUpdater] = QRegression,
        critic_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        buffer_class: Type[BaseBuffer] = ReplayBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = GaussianExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(
            start_steps=1000, scale=0.1
        ),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        device: Union[T.device, str] = "auto",
        verbose: bool = True,
        render: bool = False,
        model_path: Optional[str] = None,
        tensorboard_log_path: Optional[str] = "runs/DDPG",
    ) -> None:
        model = model or get_default_model(env)
        super().__init__(
            env,
            model,
            callbacks=callbacks,
            device=device,
            verbose=verbose,
            model_path=model_path,
            tensorboard_log_path=tensorboard_log_path,
            render=render,
        )
        self.buffer = buffer_class(
            env=env,
            buffer_size=buffer_settings.buffer_size,
            n_envs=buffer_settings.n_envs,
            device=device,
        )
        self.actor_updater = actor_updater_class(
            optimizer_class=actor_optimizer_settings.optimizer_class,
            lr=actor_optimizer_settings.learning_rate,
            max_grad=actor_optimizer_settings.max_grad,
        )
        self.critic_updater = critic_updater_class(
            optimizer_class=critic_optimizer_settings.optimizer_class,
            lr=critic_optimizer_settings.learning_rate,
            max_grad=critic_optimizer_settings.max_grad,
        )
        if explorer_settings.scale is not None:
            self.action_explorer = action_explorer_class(
                action_space=env.action_space,
                start_steps=explorer_settings.start_steps,
                scale=explorer_settings.scale,
            )
        else:
            self.action_explorer = action_explorer_class(
                action_space=env.action_space, start_steps=explorer_settings.start_steps
            )

    def _fit(self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1):
        critic_losses = np.zeros(shape=(critic_epochs))
        actor_losses = np.zeros(shape=(actor_epochs))
        # Train critic for critic_epochs
        for i in range(critic_epochs):
            trajectories = self.buffer.sample(batch_size=batch_size)
            with T.no_grad():
                next_actions = self.model.target_actor(trajectories.next_observations)
                next_q_values = self.model.target_critic(
                    trajectories.next_observations, next_actions
                )
                target_q_values = TD_zero(
                    trajectories.rewards,
                    torch_to_numpy(next_q_values),
                    trajectories.dones,
                )
            critic_log = self.critic_updater(
                self.model,
                trajectories.observations,
                target_q_values,
                trajectories.actions,
            )
            critic_losses[i] = critic_log.loss
        if self.verbose:
            self.logger.debug(f"critic losses: {critic_losses[:5], critic_losses[-5:]}")

        # Train actor for actor_epochs
        for i in range(actor_epochs):
            trajectories = self.buffer.sample(batch_size=batch_size)
            actor_log = self.actor_updater(self.model, trajectories.observations)
            actor_losses[i] = actor_log.loss
        if self.verbose:
            self.logger.debug(f"actor losses: {actor_losses[:5], actor_losses[-5:]}")

        # Update target networks
        self.model.update_targets()

        return Log(
            actor_loss=np.mean(actor_losses),
            critic_loss=np.mean(critic_losses),
        )


if __name__ == "__main__":
    import gym

    env = gym.make("Pendulum-v0")
    agent = DDPG(
        env=env,
        model=None,
        explorer_settings=ExplorerSettings(start_steps=1000, scale=0.1),
    )
    agent.fit(
        num_steps=50000,
        batch_size=64,
        critic_epochs=1,
        actor_epochs=1,
        train_frequency=("step", 1),
    )
