from typing import List, Optional, Tuple, Type

import numpy as np
import torch as T
from gym import Env

from pearll.agents import BaseAgent
from pearll.buffers import BaseBuffer, ReplayBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common import utils
from pearll.common.enumerations import FrequencyType
from pearll.common.type_aliases import Log, Observation, Trajectories
from pearll.explorers.base_explorer import BaseExplorer
from pearll.models import ModelEnv
from pearll.models.actor_critics import ActorCritic
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    OptimizerSettings,
    Settings,
)
from pearll.signal_processing import return_estimators
from pearll.updaters.critics import BaseCriticUpdater, DiscreteQRegression
from pearll.updaters.environment import BaseDeepUpdater, DeepRegression


class DynaQ(BaseAgent):
    """
    Dyna-Q model based RL algorithm.

    :param env: the gym-like environment to be used
    :param agent_model: the agent model to be used
    :param env_model: the environment model to be used
    :param td_gamma: trajectory discount factor
    :param agent_updater_class: the updater class for the agent critic
    :param agent_optimizer_settings: the settings for the agent updater
    :param obs_updater_class: the updater class for the observation function in the environment model
    :param obs_optimizer_settings: the settings for the observation updater
    :param reward_updater_class: the updater class for the reward function in the environment model
    :param reward_optimizer_settings: the settings for the reward updater
    :param done_updater_class: the updater class for the done function in the environment model
    :param done_optimizer_settings: the settings for the done updater
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
        agent_model: ActorCritic,
        env_model: ModelEnv,
        td_gamma: float = 0.99,
        agent_updater_class: Type[BaseCriticUpdater] = DiscreteQRegression,
        agent_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        obs_updater_class: BaseDeepUpdater = DeepRegression,
        obs_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        reward_updater_class: Type[BaseDeepUpdater] = DeepRegression,
        reward_optimizer_settings: OptimizerSettings = OptimizerSettings(),
        done_updater_class: Optional[Type[BaseDeepUpdater]] = None,
        done_optimizer_settings: OptimizerSettings = OptimizerSettings(
            loss_class=T.nn.BCELoss()
        ),
        buffer_class: Type[BaseBuffer] = ReplayBuffer,
        buffer_settings: BufferSettings = BufferSettings(),
        action_explorer_class: Type[BaseExplorer] = BaseExplorer,
        explorer_settings: ExplorerSettings = ExplorerSettings(start_steps=100),
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[Settings]] = None,
        logger_settings: LoggerSettings = LoggerSettings(),
        misc_settings: MiscellaneousSettings = MiscellaneousSettings(),
    ) -> None:
        super().__init__(
            env=env,
            model=agent_model,
            action_explorer_class=action_explorer_class,
            explorer_settings=explorer_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            callbacks=callbacks,
            callback_settings=callback_settings,
            misc_settings=misc_settings,
        )
        self.env_model = env_model
        self.td_gamma = td_gamma
        self.learning_rate = agent_optimizer_settings.learning_rate
        self.policy_updater = agent_updater_class(
            loss_class=agent_optimizer_settings.loss_class,
            optimizer_class=agent_optimizer_settings.optimizer_class,
            max_grad=agent_optimizer_settings.max_grad,
        )
        self.obs_updater = obs_updater_class(
            loss_class=obs_optimizer_settings.loss_class,
            optimizer_class=obs_optimizer_settings.optimizer_class,
            max_grad=obs_optimizer_settings.max_grad,
        )
        self.reward_updater = reward_updater_class(
            loss_class=reward_optimizer_settings.loss_class,
            optimizer_class=reward_optimizer_settings.optimizer_class,
            max_grad=reward_optimizer_settings.max_grad,
        )
        if self.env_model.done_fn is not None:
            self.done_updater = done_updater_class(
                loss_class=done_optimizer_settings.loss_class,
                optimizer_class=done_optimizer_settings.optimizer_class,
                max_grad=done_optimizer_settings.max_grad,
            )

        self.model_step = 0
        self.model_episode = 0

    def step_model_env(
        self, observation: Observation, num_steps: int = 1
    ) -> np.ndarray:
        """
        Step the agent in the model environment

        :param observation: the starting observation to step from
        :param num_steps: how many steps to take
        :return: the final observation after all steps have been done
        """
        self.model.eval()
        for _ in range(num_steps):
            action = self.action_explorer(self.model, observation, self.step)
            next_observation, reward, done, _ = self.env_model.step(observation, action)
            self.buffer.add_trajectory(
                observation, action, reward, next_observation, done
            )
            self.logger.debug(
                Trajectories(observation, action, reward, next_observation, done)
            )
            observation = next_observation

            if done:
                observation = self.env_model.reset()
                self.model_episode += 1
            self.model_step += 1

        return observation

    def _fit_model_env(self, batch_size: int, epochs: int = 1) -> None:
        """
        Fit the model environment

        :param batch_size: the batch size to use
        :param epochs: how many epochs to fit for
        """
        obs_loss = np.zeros(epochs)
        reward_loss = np.zeros(epochs)
        done_loss = np.zeros(epochs)
        for i in range(epochs):
            trajectories = self.buffer.sample(batch_size, dtype="torch")
            obs_update_log = self.obs_updater(
                model=self.env_model.observation_fn,
                observations=trajectories.observations,
                actions=trajectories.actions,
                targets=trajectories.next_observations,
                learning_rate=self.learning_rate,
            )
            obs_loss[i] = obs_update_log.loss
            reward_update_log = self.reward_updater(
                model=self.env_model.reward_fn,
                observations=trajectories.observations,
                actions=trajectories.actions,
                targets=trajectories.rewards,
                learning_rate=self.learning_rate,
            )
            reward_loss[i] = reward_update_log.loss
            if self.env_model.done_fn is not None:
                done_update_log = self.done_updater(
                    model=self.env_model.done_fn,
                    observations=trajectories.observations,
                    actions=trajectories.actions,
                    targets=trajectories.dones,
                    learning_rate=self.learning_rate,
                )
                done_loss[i] = done_update_log.loss

        self.logger.debug(f"obs_loss: {obs_loss.mean()}")
        self.logger.debug(f"reward_loss: {reward_loss.mean()}")
        if self.env_model.done_fn is not None:
            self.logger.debug(f"done_loss: {done_loss.mean()}")

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
                next_q_values = utils.to_numpy(next_q_values.max(dim=-1)[0])
                next_q_values = next_q_values[..., np.newaxis]
                target_q_values = return_estimators.TD_zero(
                    trajectories.rewards,
                    next_q_values,
                    trajectories.dones,
                    self.td_gamma,
                )

            updater_log = self.policy_updater(
                self.model,
                trajectories.observations,
                target_q_values,
                trajectories.actions,
                learning_rate=self.learning_rate,
            )
            critic_losses[i] = updater_log.loss

        self.model.assign_targets()

        return Log(critic_loss=np.mean(critic_losses))

    def fit(
        self,
        env_steps: int,
        plan_steps: int,
        env_batch_size: int,
        plan_batch_size: int,
        actor_epochs: int = 1,
        critic_epochs: int = 1,
        env_epochs: int = 1,
        env_train_frequency: Tuple[str, int] = ("step", 1),
        plan_train_frequency: Tuple[str, int] = ("step", 1),
        no_model_steps: int = 0,
    ) -> None:
        """
        Train the agent in the environment

        1. Collect samples in the real environment.
        2. Train the model environment on samples collected.
        3. Collect samples in the model environment.
        4. Train the agent on samples collected in both environments.

        :param env_steps: total number of real environment steps to train over
        :param plan_steps: number of model environment steps to run each planning phase
        :param env_batch_size: minibatch size for the model environment to make a single gradient descent step on
        :param plan_batch_size: minibatch size for the agent to make a single gradient descent step on
        :param actor_epochs: how many times to update the actor network in each training step
        :param critic_epochs: how many times to update the critic network in each training step
        :param env_epochs: how many times to update the model environment in each training step
        :param env_train_frequency: the number of steps or episodes to run in the real environment before running a model environment training step.
            To run every n episodes, use `("episode", n)`.
            To run every n steps, use `("step", n)`.
        :param plan_train_frequency: the number of steps or episodes to run in the model environment before running an agent training step.
            To run every n episodes, use `("episode", n)`.
            To run every n steps, use `("step", n)`.
        :param no_model_steps: number of steps to run without collecting trajectories from the model environment.
        """
        env_train_frequency = (
            FrequencyType(env_train_frequency[0].lower()),
            env_train_frequency[1],
        )
        plan_train_frequency = (
            FrequencyType(plan_train_frequency[0].lower()),
            plan_train_frequency[1],
        )

        # We can pre-calculate how many training steps to run if train frequency is in steps rather than episodes
        if env_train_frequency[0] == FrequencyType.STEP:
            env_steps = env_steps // env_train_frequency[1]
        if plan_train_frequency[0] == FrequencyType.STEP:
            plan_steps = plan_steps // plan_train_frequency[1]

        observation = self.env.reset()
        for _ in range(env_steps):
            self.logger.debug("REAL ENVIRONMENT")
            # Step for number of steps specified
            if env_train_frequency[0] == FrequencyType.STEP:
                observation = self.step_env(
                    observation=observation, num_steps=env_train_frequency[1]
                )
            # Step for number of episodes specified
            elif env_train_frequency[0] == FrequencyType.EPISODE:
                start_episode = self.episode
                end_episode = start_episode + env_train_frequency[1]
                while self.episode != end_episode:
                    observation = self.step_env(observation=observation)
                if self.step >= env_steps:
                    break

            if self.done:
                break

            # Update the environment model
            self._fit_model_env(batch_size=env_batch_size, epochs=env_epochs)

            if self.step < no_model_steps:
                # Update the agent model
                self.model.train()
                train_log = self._fit(
                    batch_size=plan_batch_size,
                    actor_epochs=actor_epochs,
                    critic_epochs=critic_epochs,
                )
                self.model.update_global()
                self.logger.add_train_log(train_log)
            else:
                self.logger.debug("MODEL ENVIRONMENT")
                # Plan for number of steps specified
                model_obs = self.env_model.reset()
                for _ in range(plan_steps):
                    # Step for number of steps specified
                    if plan_train_frequency[0] == FrequencyType.STEP:
                        model_obs = self.step_model_env(
                            observation=model_obs, num_steps=plan_train_frequency[1]
                        )
                    # Step for number of episodes specified
                    elif plan_train_frequency[0] == FrequencyType.EPISODE:
                        start_episode = self.model_episode
                        end_episode = start_episode + plan_train_frequency[1]
                        while self.model_episode != end_episode:
                            observation = self.step_model_env(observation=observation)
                        if self.model_step >= plan_steps:
                            break

                    # Update the agent model
                    self.model.train()
                    train_log = self._fit(
                        batch_size=plan_batch_size,
                        actor_epochs=actor_epochs,
                        critic_epochs=critic_epochs,
                    )
                    self.model.update_global()
                    self.logger.add_train_log(train_log)

                self.buffer.reset()
