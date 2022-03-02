from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as T
from gym import Env
from gym.vector import VectorEnv

from pearll.buffers.base_buffer import BaseBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.enumerations import FrequencyType
from pearll.common.logging_ import Logger
from pearll.common.type_aliases import Log, Observation, Tensor, Trajectories
from pearll.common.utils import get_device, set_seed
from pearll.explorers.base_explorer import BaseExplorer
from pearll.models.actor_critics import ActorCritic
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    Settings,
)


class BaseAgent(ABC):
    """
    The BaseAgent class is given to handle all the stuff around either RL or EC algorithms.
    It's recommended to inherit this class when implementing your own agents. You'll need
    to implement the _fit() abstract method and override the __init__ to add updaters along
    with their settings.

    See the example agents already done for guidance and settings.py for settings objects
    that can be used.

    :param env: the gym-like environment to be used
    :param model: the neural network model
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param action_explorer_class: the explorer class for random search at beginning of training and
        adding noise to actions
    :param explorer settings: settings for the action explorer
    :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
    :param callback_settings: settings for callbacks
    :param logger_settings: settings for the logger
    :param misc_settings: settings for miscellaneous parameters
    """

    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        buffer_class: Type[BaseBuffer] = BaseBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[Settings]] = None,
        logger_settings: LoggerSettings = LoggerSettings(),
        misc_settings: MiscellaneousSettings = MiscellaneousSettings(),
    ) -> None:
        self.env = env
        self.model = model
        self.render = misc_settings.render
        explorer_settings = explorer_settings.filter_none()
        self.action_explorer = action_explorer_class(
            action_space=env.action_space, **explorer_settings
        )
        buffer_settings = buffer_settings.filter_none()
        self.buffer = buffer_class(
            env=env, device=misc_settings.device, **buffer_settings
        )
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
        self.log_frequency = (
            FrequencyType(logger_settings.log_frequency[0].lower()),
            logger_settings.log_frequency[1],
        )
        if callbacks is not None:
            assert len(callbacks) == len(
                callback_settings
            ), "There should be a CallbackSetting object for each callback"
            callback_settings = [setting.filter_none() for setting in callback_settings]
            self.callbacks = [
                callback(logger=self.logger, model=self.model, **settings)
                for callback, settings in zip(callbacks, callback_settings)
            ]
        else:
            self.callbacks = None

        device = get_device(misc_settings.device)
        self.logger.info(f"Using device {device}")

        if misc_settings.seed is not None:
            self.logger.info(f"Using seed {misc_settings.seed}")
            set_seed(misc_settings.seed, self.env)

    def predict(self, observations: Union[Tensor, Dict[str, Tensor]]) -> T.Tensor:
        """Run the agent actor model"""
        self.model.eval()
        return self.model.predict(observations)

    def action_distribution(
        self, observations: Union[Tensor, Dict[str, Tensor]]
    ) -> T.distributions.Distribution:
        """Get the policy distribution given an observation"""
        self.model.eval()
        return self.model.predict_distribution(observations)

    def critic(
        self,
        observations: Union[Tensor, Dict[str, Tensor]],
        actions: Optional[Tensor] = None,
    ) -> T.Tensor:
        """Run the agent critic model"""
        self.model.eval()
        return self.model.predict_critic(observations, actions)

    def dump_log(self) -> None:
        """
        Write and reset the logger
        """
        self.logger.write_log(self.step)
        self.logger.reset_log()

    def step_env(self, observation: Observation, num_steps: int = 1) -> np.ndarray:
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
            next_observation, reward, done, _ = self.env.step(action)
            self.buffer.add_trajectory(
                observation, action, reward, next_observation, done
            )
            self.logger.debug(
                f"{Trajectories(observation, action, reward, next_observation, done)}"
            )
            # Add reward to current episode log
            self.logger.add_reward(reward)
            observation = next_observation

            # If all environment episodes are done, reset and check if we should dump the log
            if self.logger.check_episode_done(done):
                observation = self.env.reset()
                if self.log_frequency[0] == FrequencyType.EPISODE:
                    if self.episode % self.log_frequency[1] == 0:
                        self.dump_log()
                self.episode += 1

            if self.log_frequency[0] == FrequencyType.STEP:
                if self.step % self.log_frequency[1] == 0:
                    self.dump_log()
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
            FrequencyType(train_frequency[0].lower()),
            train_frequency[1],
        )
        # We can pre-calculate how many training steps to run if train_frequency is in steps rather than episodes
        if train_frequency[0] == FrequencyType.STEP:
            num_steps = num_steps // train_frequency[1]

        observation = self.env.reset()
        for step in range(num_steps):
            # Always fill buffer with enough samples for first training step
            if step == 0:
                observation = self.step_env(
                    observation=observation, num_steps=batch_size
                )
            # Step for number of steps specified
            elif train_frequency[0] == FrequencyType.STEP:
                observation = self.step_env(
                    observation=observation, num_steps=train_frequency[1]
                )
            # Step for number of episodes specified
            elif train_frequency[0] == FrequencyType.EPISODE:
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
            self.model.update_global()
            self.logger.add_train_log(train_log)
