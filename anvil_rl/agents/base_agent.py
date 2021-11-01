import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch as T
from gym import Env, spaces
from torch.utils.tensorboard import SummaryWriter

from anvil_rl.callbacks.base_callback import BaseCallback
from anvil_rl.common.enumerations import TrainFrequencyType
from anvil_rl.common.type_aliases import Log, Tensor
from anvil_rl.common.utils import get_device
from anvil_rl.models.actor_critics import ActorCritic


class BaseAgent(ABC):
    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        model_path: Optional[str] = None,
        device: Union[str, T.device] = "auto",
        tensorboard_log_path: Optional[str] = None,
        verbose: bool = True,
        render: bool = False,
    ) -> None:
        """
        The BaseAgent class is given to handle all the stuff around the actual RL algorithm.
        It's recommended to inherit this class when implementing your own agents. You'll need
        to implement the _fit() abstract method and override the __init__ to add the actor and
        critic updaters and their settings as well as the action explorer with its settings.

        See the example algorithms already done for guidance and common/type_aliases.py for
        settings objects that can be used.

        :param env: the gym-like environment to be used
        :param model: the neural network model
        :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
        :param model_path: optional model path to load from
        :param device: device to run on, accepts "auto", "cuda" or "cpu"
        :param tensorboard_log_path: path to store the tensorboard log
        :param verbose: whether to display at all or not
        :param render: whether to render the environment or not
        """

        self.env = env
        self.model = model
        self.verbose = verbose
        self.model_path = model_path
        self.callbacks = callbacks
        self.render = render
        self.action_explorer = None
        self.step = 0
        self.episode = 0
        self.logger = self.get_logger()

        device = get_device(device)
        if verbose:
            self.logger.info(f"Using device {device}")

        self.buffer = None
        self._reset_episode_log()
        self.writer = SummaryWriter(tensorboard_log_path)
        # Load the model if a path is given
        if self.model_path is not None:
            self.load(model_path)

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler("agent.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

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

    def _reset_episode_log(self) -> None:
        self.episode_actor_losses = []
        self.episode_critic_losses = []
        self.episode_kl_divergences = []
        self.episode_entropies = []
        self.episode_rewards = []

    def _write_log(self, log: Log, step: int) -> None:
        """Write a log to tensorboard and python logging"""
        self.writer.add_scalar("Reward/episode_reward", log.reward, step)
        self.writer.add_scalar("Loss/actor_loss", log.actor_loss, step)
        self.writer.add_scalar("Loss/critic_loss", log.critic_loss, step)
        if log.kl_divergence is not None:
            self.writer.add_scalar("Metrics/kl_divergence", log.kl_divergence, step)
        if log.entropy is not None:
            self.writer.add_scalar("Metrics/entropy", log.entropy, step)

        if self.verbose:
            self.logger.info(f"{step}: {log}")

    def predict(self, observations: Tensor) -> T.Tensor:
        """Run the agent actor model"""
        self.model.eval()
        return self.model(observations)

    def get_action_distribution(
        self, observations: Tensor
    ) -> T.distributions.Distribution:
        """Get the policy distribution given an observation"""
        self.model.eval()
        return self.model.get_action_distribution(observations)

    def critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Run the agent critic model"""
        self.model.eval()
        return self.model.critic(observations, actions)

    def _process_action(self, action: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        Process the action taken from the action explorer ready for the `env.step()` method
        and buffer storage.

        :param action: the action taken from the action explorer
        :return: the processed action
        """
        if isinstance(self.env.action_space, spaces.Discrete) and not isinstance(
            action, int
        ):
            action = action.item()
        return action

    def step_env(self, observation: np.ndarray, num_steps: int = 1) -> np.ndarray:
        """
        Step the agent in the environment

        :param observation: the starting observation to step from
        :param num_steps: how many steps to take
        :return: the final observation after all steps have been done
        """
        self.model.eval()
        for _ in range(num_steps):
            if self.render:
                self.env.render()
            action = self.action_explorer(self.model, observation, self.step)
            action = self._process_action(action)
            next_observation, reward, done, _ = self.env.step(action)
            self.buffer.add_trajectory(
                observation, action, reward, next_observation, done
            )

            self.episode_rewards.append(reward)
            if self.verbose:
                self.logger.debug(f"ACTION: {action}")
                self.logger.debug(f"REWARD: {reward}")

            if done:
                observation = self.env.reset()
                episode_log = Log(
                    reward=np.sum(self.episode_rewards),
                )
                if self.episode_actor_losses:
                    episode_log.actor_loss = np.mean(self.episode_actor_losses)
                if self.episode_critic_losses:
                    episode_log.critic_loss = np.mean(self.episode_critic_losses)
                if self.episode_kl_divergences:
                    episode_log.kl_divergence = np.mean(self.episode_kl_divergences)
                if self.episode_entropies:
                    episode_log.entropy = np.mean(self.episode_entropies)
                self._write_log(episode_log, self.step)
                self._reset_episode_log()
                self.episode += 1
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
        train_frequency: Tuple[str, int] = ("step", 1),
    ) -> None:
        """
        Train the agent in the environment

        :param num_steps: total number of environment steps to train over
        :param batch_size: minibatch size to make a single gradient descent step on
        :param actor_epochs: how many times to update the actor network in each training step
        :param critic_epochs: how many times to update the critic network in each training step
        :param train_frequency: the number of steps or episodes to run before running a training step.
            To run every n episodes, use `("episode", n)`.
            To run every n steps, use `("step", n)`.
        """
        train_frequency = (
            TrainFrequencyType(train_frequency[0].lower()),
            train_frequency[1],
        )
        # We can pre-calculate how many training steps to run if train_frequency is in steps rather than episodes
        if train_frequency[0] == TrainFrequencyType.STEP:
            num_steps = num_steps // train_frequency[1]

        observation = self.env.reset()
        for step in range(num_steps):
            # Always fill buffer with enough samples for first training step
            if step == 0:
                observation = self.step_env(
                    observation=observation, num_steps=batch_size
                )
            # Step for number of steps specified
            elif train_frequency[0] == TrainFrequencyType.STEP:
                observation = self.step_env(
                    observation=observation, num_steps=train_frequency[1]
                )
            # Step for number of episodes specified
            elif train_frequency[0] == TrainFrequencyType.EPISODE:
                start_episode = self.episode
                end_episode = start_episode + train_frequency[1]
                while self.episode != end_episode:
                    observation = self.step_env(observation=observation)
                if self.step >= num_steps:
                    break

            self.model.train()
            train_log = self._fit(
                batch_size=batch_size,
                actor_epochs=actor_epochs,
                critic_epochs=critic_epochs,
            )

            self.episode_actor_losses.append(train_log.actor_loss)
            self.episode_critic_losses.append(train_log.critic_loss)
            if train_log.entropy is not None:
                self.episode_entropies.append(train_log.entropy)
            if train_log.kl_divergence is not None:
                self.episode_kl_divergences.append(train_log.kl_divergence)
