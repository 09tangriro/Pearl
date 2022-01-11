from typing import Union

import numpy as np
from gym import spaces

from pearll.common.type_aliases import Observation
from pearll.explorers.base_explorer import BaseExplorer
from pearll.models.actor_critics import Actor, ActorCritic


class GaussianExplorer(BaseExplorer):
    """
    Add Gaussian noise to the actions

    :param action_space: action space
    :param scale: std of the Gaussian noise
    :param start_steps: the fist n steps to uniformally sample actions
    """

    def __init__(
        self,
        action_space: spaces.Space,
        scale: float = 0.1,
        start_steps: int = 20000,
    ) -> None:
        super().__init__(action_space, start_steps=start_steps)
        self.scale = scale

    def __call__(
        self, model: Union[Actor, ActorCritic], observation: Observation, step: int
    ) -> np.ndarray:
        """
        Get an action for the given observation in training.

        :param model: the model to use
        :param observation: the observation
        :param step: the current training step
        """
        actions = super().__call__(model, observation, step)
        if step >= self.start_steps:
            noises = np.random.normal(loc=0.0, scale=self.scale, size=self.action_shape)
            actions = actions + noises
            actions = np.clip(actions, self.action_range[0], self.action_range[1])
        return actions
