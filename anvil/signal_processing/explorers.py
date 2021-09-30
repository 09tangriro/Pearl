from typing import Union

import numpy as np
from gym import Space

from anvil.models.actor_critics import Actor, ActorCritic


class BaseExplorer(object):
    """
    Base explorer class that should be used when no extra noise is added to the actions/a stochastic actor is used

    :param actor: actor or actor critic network
    :param action_space: action space
    :param start_steps: the fist n steps to uniformally sample actions
    """

    def __init__(
        self,
        actor: Union[Actor, ActorCritic],
        action_space: Space,
        start_steps: int = 20000,
    ) -> None:
        self.start_steps = start_steps
        self.actor = actor
        self.action_space = action_space
        self.action_size = action_space.shape[0]

    def __call__(self, observation: np.ndarray, step: int) -> np.ndarray:
        if step > self.start_steps:
            actions = self.actor(observation)
        else:
            shape = (len(observation), self.action_size)
            actions = np.random.uniform(
                low=self.action_space.low, high=self.action_space.high, shape=shape
            )
        return actions


class GaussianExplorer(BaseExplorer):
    def __init__(
        self,
        actor: Union[Actor, ActorCritic],
        action_space: Space,
        scale: float = 0.1,
        start_steps: int = 20000,
    ) -> None:
        """
        Add Gaussian noise to the actions

        :param actor: actor or actor critic network
        :param action_space: action space
        :param scale: std of the Gaussian noise
        :param start_steps: the fist n steps to uniformally sample actions
        """
        super().__init__(actor, action_space, start_steps=start_steps)
        self.scale = scale

    def __call__(self, observation: np.ndarray, step: int) -> np.ndarray:
        actions = super().__call__(observation, step)
        if step > self.start_steps:
            noises = np.random.normal(scale=self.scale, size=actions.shape)
            actions = (actions + noises).astype(np.float32)
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions
