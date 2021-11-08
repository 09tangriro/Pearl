from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch as T
from gym import Env, spaces
from gym.vector import VectorEnv

from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.callbacks.base_callback import BaseCallback
from anvilrl.common.enumerations import TrainFrequencyType
from anvilrl.common.logging import Logger
from anvilrl.common.type_aliases import Log, Tensor
from anvilrl.common.utils import filter_dataclass_by_none, get_device, numpy_to_torch
from anvilrl.explorers.base_explorer import BaseExplorer
from anvilrl.models.actor_critics import Actor, ActorCritic
from anvilrl.settings import (
    BufferSettings,
    CallbackSettings,
    ExplorerSettings,
    LoggerSettings,
)


class BaseDeepAgent(ABC):
    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(),
        buffer_class: BaseBuffer = BaseBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        logger_settings: LoggerSettings = LoggerSettings(),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[CallbackSettings]] = None,
        device: Union[str, T.device] = "auto",
        render: bool = False,
    ) -> None:
        """
        The BaseDeepAgent class is given to handle all the stuff around the actual Deep RL algorithm.
        It's recommended to inherit this class when implementing your own Deep RL agent. You'll need
        to implement the _fit() abstract method and override the __init__ to add buffer, updaters
        and explorers along with their respective settings.

        See the example algorithms already done for guidance and settings.py for settings objects
        that can be used.

        :param env: the gym-like environment to be used
        :param model: the neural network model
        :param action_explorer_class: the explorer class for random search at beginning of training and
            adding noise to actions
        :param explorer settings: settings for the action explorer
        :param buffer_class: the buffer class for storing and sampling trajectories
        :param buffer_settings: settings for the buffer
        :param logger_settings: settings for the logger
        :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
        :param callback_settings: settings for callbacks
        :param device: device to run on, accepts "auto", "cuda" or "cpu"
        :param render: whether to render the environment or not
        """

        self.env = env
        self.model = model
        self.render = render
        explorer_settings = filter_dataclass_by_none(explorer_settings)
        self.action_explorer = action_explorer_class(
            action_space=env.action_space, **explorer_settings
        )
        buffer_settings = filter_dataclass_by_none(buffer_settings)
        self.buffer = buffer_class(env=env, device=device, **buffer_settings)
        self.step = 0
        self.episode = 0
        self.done = False  # Flag terminate training
        self.logger = Logger(
            tensorboard_log_path=logger_settings.tensorboard_log_path,
            file_handler_level=logger_settings.file_handler_level,
            stream_handler_level=logger_settings.stream_handler_level,
            verbose=logger_settings.verbose,
            num_envs=env.num_envs if isinstance(env, VectorEnv) else 1,
        )
        if callbacks is not None:
            assert len(callbacks) == len(
                callback_settings
            ), "There should be a CallbackSetting object for each callback"
            callback_settings = [
                filter_dataclass_by_none(setting) for setting in callback_settings
            ]
            self.callbacks = [
                callback(self.logger, self.model, **asdict(settings))
                for callback, settings in zip(callbacks, callback_settings)
            ]
        else:
            self.callbacks = None

        device = get_device(device)
        self.logger.info(f"Using device {device}")

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
            if isinstance(self.env, VectorEnv):
                action = [
                    self.action_explorer(self.model, env_obs, self.step)
                    for env_obs in observation
                ]
                action = np.array([self._process_action(a) for a in action])
            else:
                action = self.action_explorer(self.model, observation, self.step)
                action = self._process_action(action)
            next_observation, reward, done, _ = self.env.step(action)
            self.buffer.add_trajectory(
                observation, action, reward, next_observation, done
            )
            self.logger.add_reward(reward)
            self.logger.debug(f"ACTION: {action}")
            self.logger.debug(f"REWARD: {reward}")
            done_indices = np.where(done)[0]

            if isinstance(self.env, VectorEnv):
                not_done_indices = np.where(~done)[0]
                observation[done_indices] = self.env.reset()[done_indices]
                observation[not_done_indices] = next_observation[not_done_indices]
            else:
                observation = self.env.reset() if done else next_observation

            # For multiple environments, we keep track of individual episodes as they finish
            self.logger.epsiode_dones[done_indices] = True
            # If all environment episodes are done, we write an episode log and reset it.
            if all(self.logger.epsiode_dones):
                self.logger.write_episode_log(self.step)
                self.logger.reset_episode_log()
                self.episode += 1

            if self.callbacks is not None:
                if not all(
                    [callback.on_step(self.step) for callback in self.callbacks]
                ):
                    self.done = True
                    break
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

            if self.done:
                break

            self.model.train()
            train_log = self._fit(
                batch_size=batch_size,
                actor_epochs=actor_epochs,
                critic_epochs=critic_epochs,
            )

            self.logger.add_train_log(train_log)


class BaseSearchAgent(ABC):
    """TBD"""

    def __init__(
        self,
        env: VectorEnv,
        model: Actor,
        population_size: int,
        population_std: float,
        logger_settings: LoggerSettings = LoggerSettings(),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[CallbackSettings]] = None,
        device: Union[str, T.device] = "auto",
        render: bool = False,
    ) -> None:
        self.env = env
        self.model = model
        self.population_size = population_size
        self.population_std = population_std
        self.render = render
        self.buffer = None
        self.step = 0
        self.episode = 0
        self.done = False  # Flag terminate training
        # Keep track of which individuals have completed an episode
        self.epsiode_dones = np.array([False for _ in range(population_size)])
        self.logger = Logger(
            tensorboard_log_path=logger_settings.tensorboard_log_path,
            file_handler_level=logger_settings.file_handler_level,
            stream_handler_level=logger_settings.stream_handler_level,
            verbose=logger_settings.verbose,
            num_envs=env.num_envs,
        )
        if callbacks is not None:
            assert len(callbacks) == len(
                callback_settings
            ), "There should be a CallbackSetting object for each callback"
            self.callbacks = [
                callback(self.logger, self.model, **asdict(settings))
                for callback, settings in zip(callbacks, callback_settings)
            ]
        else:
            self.callbacks = None

        device = get_device(device)
        self.logger.info(f"Using device {device}")

    def predict(self, observations: Tensor) -> T.Tensor:
        """Run the agent actor model"""
        return self.model(observations)

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

    def step_env(
        self, models: List[Actor], observation: np.ndarray, num_steps: int = 1
    ) -> np.ndarray:
        """
        Step the agent in the environment

        :param observation: the starting observation to step from
        :param num_steps: how many steps to take
        :return: the final observation after all steps have been done
        """

        for _ in range(num_steps):
            if self.render:
                self.env.render()
            action = [model(observation[i]) for i, model in enumerate(models)]
            action = [self._process_action(a) for a in action]
            next_observation, reward, done, _ = self.env.step(action)
            self.buffer.add_trajectory(
                observation, action, reward, next_observation, done
            )

            self.logger.add_reward(reward)
            self.logger.debug(f"ACTION: {action}")
            self.logger.debug(f"REWARD: {reward}")

            done_indices = np.where(done)[0]
            not_done_indices = np.where(~done)[0]
            observation[done_indices] = self.env.reset()[done_indices]
            observation[not_done_indices] = next_observation[not_done_indices]

            self.logger.epsiode_dones[done_indices] = True
            if all(self.logger.epsiode_dones):
                self.logger.write_episode_log(self.step)
                self.logger.reset_episode_log()
                self.episode += 1
            self.step += 1
        return observation

    @staticmethod
    def _model_to_vector(model: Actor) -> np.ndarray:
        weights, biases = [p for p in model.parameters() if p.requires_grad]
        parameter_vector = T.cat((T.flatten(weights), biases))
        return numpy_to_torch(parameter_vector)

    @staticmethod
    def _vector_to_model(vector: np.ndarray) -> Actor:
        raise NotImplementedError()

    def initialize_population(self) -> np.ndarray:
        mean = self._model_to_vector(self.model)
        normal_dist = np.random.randn(self.population_size, len(mean))
        return mean + self.population_std * normal_dist

    @abstractmethod
    def _fit(
        self, population_size: int, population_std: float, learning_rate: float
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
        train_frequency: Tuple[str, int] = ("step", 1),
    ) -> None:
        """
        Train the agent in the environment

        :param num_steps: total number of environment steps to train over
        :param population_size: how many agents to compare in each iteration
        :param actor_epochs: how many times to update the actor network in each training step
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

        self.population = self.initialize_population()
        observation = self.env.reset()
        for _ in range(num_steps):
            models = [
                self._vector_to_model(individual) for individual in self.population
            ]
            # Step for number of steps specified
            if train_frequency[0] == TrainFrequencyType.STEP:
                observation = self.step_env(
                    models, observation=observation, num_steps=train_frequency[1]
                )
            # Step for number of episodes specified
            elif train_frequency[0] == TrainFrequencyType.EPISODE:
                start_episode = self.episode
                end_episode = start_episode + train_frequency[1]
                while self.episode != end_episode:
                    observation = self.step_env(models, observation=observation)
                if self.step >= num_steps:
                    break

            if self.done:
                break

            train_log = self._fit(
                population_size=self.population_size,
                population_std=self.population_std,
                learning_rate=1,
            )

            self.logger.add_train_log(train_log)
