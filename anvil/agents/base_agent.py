import os
from abc import ABC, abstractmethod
from logging import INFO, Logger
from typing import List, Optional, Type, Union

import numpy as np
import torch as T
from gym import Env
from torch.utils.tensorboard import SummaryWriter

from anvil.buffers.base_buffer import BaseBuffer
from anvil.buffers.rollout_buffer import RolloutBuffer
from anvil.callbacks.base_callback import BaseCallback
from anvil.common.type_aliases import Log, Tensor
from anvil.common.utils import get_device, torch_to_numpy
from anvil.models.actor_critics import ActorCritic


class BaseAgent(ABC):
    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        buffer_class: Type[BaseBuffer],
        buffer_size: int,
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        device: Union[T.device, str] = "auto",
        verbose: bool = True,
        model_path: Optional[str] = None,
        tensorboard_log_path: Optional[str] = None,
        n_envs: int = 1,
    ) -> None:
        self.env = env
        self.model = model
        self.verbose = verbose
        self.model_path = model_path
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.callbacks = callbacks
        self.action_explorer = None
        self.device = get_device(device)
        self.step = 0

        self.buffer = buffer_class(
            buffer_size=buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            n_envs=n_envs,
            device=device,
        )

        self.logger = Logger(__name__, level=INFO)
        self.writer = SummaryWriter(tensorboard_log_path)
        # Load the model if a path is given
        if self.model_path is not None:
            self.load(model_path)

    def save(self, path: str):
        """Save the model"""
        path = path + ".pt"
        if self.verbose:
            self.logger.info(f"Saving weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load the model"""
        path = path + ".pt"
        if self.verbose:
            self.logger.info(f"Loading weights from {path}")
        try:
            self.model.load_state_dict(T.load(path))
        except FileNotFoundError:
            if self.verbose:
                self.logger.info(
                    "File not found, assuming no model dict was to be loaded"
                )

    def _write_log(self, log: Log, step: int) -> None:
        """Write a log to tensorboard and python logging"""
        self.writer.add_scalar("reward", log.reward, step)
        self.writer.add_scalar("actor_loss", log.actor_loss, step)
        self.writer.add_scalar("critic_loss", log.critic_loss, step)
        if log.kl_divergence is not None:
            self.writer.add_scalar("kl_divergence", log.kl_divergence, step)
        if log.entropy is not None:
            self.writer.add_scalar("entropy", log.entropy, step)

        self.logger.info(f"{step}: {log}")

    def predict(self, observations: Tensor) -> T.Tensor:
        """Run the agent actor model"""
        return self.model(observations)

    def get_action_distribution(
        self, observations: Tensor
    ) -> T.distributions.Distribution:
        """Get the policy distribution given an observation"""
        return self.model.get_action_distribution(observations)

    def critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Run the agent critic model"""
        return self.model.critic(observations, actions)

    def step_env(self, observation: np.ndarray, num_steps: int = 1) -> np.ndarray:
        """
        Step the agent in the environment

        :param observation: the starting observation to step from
        :param num_steps: how many steps to take
        :return: the final observation after all steps have been done
        """
        for _ in range(num_steps):
            if self.action_explorer is not None:
                action = self.action_explorer(observation, self.step)
            else:
                action = self.model(observation)
            numpy_action = torch_to_numpy(action)
            next_observation, reward, done, _ = self.env.step(numpy_action)
            self.buffer.add_trajectory(
                observation, numpy_action, reward, next_observation, done
            )
            if done:
                observation = self.env.reset()
            else:
                observation = next_observation
            self.step += 1
        return observation

    @abstractmethod
    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        """
        Train the agent in the environment

        :param batch_size: minibatch size to make a single gradient descent step on
        :param actor_epochs: how many times to update the actor network in each training step
        :param critic_epochs: how many times to update the critic network in each training step
        :return: a Log object with training diagnostic info
        """

    def fit(
        self,
        num_steps: int,
        batch_size: int,
        actor_epochs: int = 1,
        critic_epochs: int = 1,
    ) -> None:
        """
        Train the agent in the environment

        :param num_steps: total number of environment steps to train over
        :param batch_size: minibatch size to make a single gradient descent step on
        :param actor_epochs: how many times to update the actor network in each training step
        :param critic_epochs: how many times to update the critic network in each training step
        """
        # Assume RolloutBuffer is used with on-policy agents, so translate env steps to training steps
        if isinstance(self.buffer, RolloutBuffer):
            num_steps = num_steps // self.buffer_size

        observation = self.env.reset()
        for step in range(num_steps):
            # Always fill buffer with enough samples for first training step
            if step == 0:
                observation = self.step_env(
                    observation=observation, num_steps=batch_size
                )
            # For on-policy, fill buffer and get minibatch samples over epochs
            elif isinstance(self.buffer, RolloutBuffer):
                observation = self.step_env(
                    observation=observation, num_steps=self.buffer_size
                )
            # For off-policy only a single step is done since old samples can be reused
            else:
                observation = self.step_env(observation=observation)
            log = self._fit(
                batch_size=batch_size,
                actor_epochs=actor_epochs,
                critic_epochs=critic_epochs,
            )
            log.reward = np.mean(self.buffer.last(batch_size=batch_size).rewards)
            self._write_log(log, step)
