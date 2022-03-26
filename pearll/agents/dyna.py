from typing import List, Optional, Tuple, Type

import numpy as np
import torch as T
from gym import Env

from pearll.agents import BaseAgent
from pearll.buffers import BaseBuffer, ReplayBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common import utils
from pearll.common.enumerations import FrequencyType
from pearll.common.type_aliases import Log, Observation
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
        explorer_settings: ExplorerSettings = ExplorerSettings(),
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
            observation = next_observation

            # If all environment episodes are done, reset and check if we should dump the log
            if self.logger.check_episode_done(done):
                observation = self.env_model.reset()
                self.model_episode += 1
            self.model_step += 1

        return observation

    def _fit_model_env(self, batch_size: int, epochs: int = 1) -> None:
        for _ in range(epochs):
            trajectories = self.buffer.sample(batch_size, dtype="torch")
            obs_update_log = self.obs_updater(
                model=self.env_model.observation_fn,
                observations=trajectories.observations,
                actions=trajectories.actions,
                targets=trajectories.next_observations,
                learning_rate=self.learning_rate,
            )
            self.logger.debug(f"Observation update log: {obs_update_log}")
            reward_update_log = self.reward_updater(
                model=self.env_model.reward_fn,
                observations=trajectories.observations,
                actions=trajectories.actions,
                targets=trajectories.rewards,
                learning_rate=self.learning_rate,
            )
            self.logger.debug(f"Reward update log: {reward_update_log}")
            if self.env_model.done_fn is not None:
                done_update_log = self.done_updater(
                    model=self.env_model.done_fn,
                    observations=trajectories.observations,
                    actions=trajectories.actions,
                    targets=trajectories.dones,
                    learning_rate=self.learning_rate,
                )
                self.logger.debug(f"Done update log: {done_update_log}")

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
    ) -> None:
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
