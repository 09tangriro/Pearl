from typing import Union

import numpy as np
from gym import spaces

from pearll.common.type_aliases import Observation
from pearll.common.utils import get_space_range, get_space_shape, to_numpy
from pearll.models.actor_critics import Actor, ActorCritic
from pearll.models.utils import get_mlp_size


class BaseExplorer(object):
    """
    Base explorer class that should be used when no extra noise is added to the actions/a stochastic actor is used

    :param action_space: action space
    :param start_steps: the fist n steps to uniformally sample actions
    """

    def __init__(
        self,
        action_space: spaces.Space,
        start_steps: int = 20000,
    ) -> None:
        self.start_steps = start_steps
        self.action_space = action_space
        self.action_shape = get_space_shape(action_space)
        self.action_size = get_mlp_size(get_space_shape(action_space))
        self.action_range = get_space_range(action_space)

    def __call__(
        self, model: Union[Actor, ActorCritic], observation: Observation, step: int
    ) -> np.ndarray:
        """
        Get an action for the given observation in training.

        :param model: the model to use
        :param observation: the observation
        :param step: the current training step
        """
        if step >= self.start_steps:
            action = to_numpy(model(observation))
            action = np.clip(action, self.action_range[0], self.action_range[1])
        else:
            action = self.action_space.sample()

        return action
