from typing import Any, Optional, Tuple

from gym import Env, Space

from pearll.common import utils
from pearll.common.type_aliases import DoneFunc, ObservationFunc, RewardFunc


class ModelEnv(Env):
    """
    Model the environment, M(S, A) -> R, S'

    :param reward_fn: reward_fn(S, A) -> R
    :param observation_fn: observation_fn(S, A) -> S'
    :param done_fn: optional done_fn(S, A) -> done
    :param reset_space: observation space sampled to reset the environment
    """

    def __init__(
        self,
        reward_fn: RewardFunc,
        observation_fn: ObservationFunc,
        done_fn: Optional[DoneFunc],
        reset_space: Space,
    ) -> None:
        self.reward_fn = reward_fn
        self.observation_fn = observation_fn
        self.done_fn = done_fn
        self.reset_space = reset_space

    def step(self, observation: Any, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Step in the environment

        :param observation: observation
        :param action: action
        :return: observation, reward, done, info
        """
        next_observation = self.observation_fn(observation, action)
        reward = self.reward_fn(observation, action)
        done = False if self.done_fn is None else self.done_fn(observation, action)
        return utils.to_numpy(next_observation), reward, done, {}

    def reset(self) -> Any:
        """
        Reset the environment

        :return: observation
        """
        return self.reset_space.sample()
