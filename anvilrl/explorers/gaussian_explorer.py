from typing import Union

import numpy as np
from gym import spaces

from anvilrl.common.type_aliases import Observation
from anvilrl.explorers.base_explorer import BaseExplorer
from anvilrl.models.actor_critics import Actor, ActorCritic


class GaussianExplorer(BaseExplorer):
    def __init__(
        self,
        action_space: spaces.Space,
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
        super().__init__(action_space, start_steps=start_steps)
        self.scale = scale

    def __call__(
        self, actor: Union[Actor, ActorCritic], observation: Observation, step: int
    ) -> np.ndarray:
        actions = super().__call__(actor, observation, step)
        if step >= self.start_steps:
            noises = np.random.normal(loc=0.0, scale=self.scale, size=self.action_size)
            actions = actions + noises
            actions = np.clip(actions, self.action_range[0], self.action_range[1])
        return actions
