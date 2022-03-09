from typing import Any, Optional, Tuple

import numpy as np
from gym import Env, Space, spaces

from pearll.common import utils
from pearll.models.actor_critics import Actor, Critic


class ModelEnv(Env):
    """
    Neural network approximation for the environment, M(S, A) -> R, S'
    The existing Critic object can be reused here for the reward model.
    The existing Actor object can be reused here for the observation model.

    :param reward_model: reward_model(S, A) -> R
    :param observation_model: observation_model(S, A) -> S'
    :param observation_space: observation space
    :param reward_space: reward space
    :param observation_map: optinal map for network output to observation space
    :param reward_map: optional map for network output to reward space
    """

    def __init__(
        self,
        reward_model: Critic,
        observation_model: Actor,
        observation_space: Space,
        reward_space: Space,
        observation_map: Optional[dict] = None,
        reward_map: Optional[dict] = None,
    ) -> None:
        self.reward_model = reward_model
        self.observation_model = observation_model
        self.observation_space = observation_space
        self.reward_space = reward_space
        self.observation_map = observation_map
        self.reward_map = reward_map

    def _process_obs(self, observation: np.ndarray) -> Any:
        """Process discrete observations"""
        if isinstance(self.observation_space, spaces.Discrete):
            if self.observation_map is None:
                return np.argmax(observation)
            return self.observation_map[np.argmax(observation)]

        elif isinstance(self.observation_space, spaces.MultiDiscrete):
            return self.observation_map[np.argmax(observation)]

        return observation

    def _process_reward(self, reward: np.ndarray) -> Any:
        """Process discrete rewards"""
        if isinstance(self.reward_space, spaces.Discrete):
            if self.reward_map is None:
                return np.argmax(reward)
            return self.reward_map[np.argmax(reward)]

        return reward

    def step(self, observation: Any, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Step in the environment

        :param observation: observation
        :param action: action
        :return: observation, reward, done, info
        """
        next_observation = self.observation_model(observation, action)
        reward = self.reward_model(observation, action)
        next_observation, reward = utils.to_numpy(next_observation, reward)
        return (
            self._process_obs(next_observation),
            self._process_reward(reward),
            False,
            {},
        )

    def reset(self) -> Any:
        """
        Reset the environment

        :return: observation
        """
        return self.observation_space.sample()
