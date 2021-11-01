from typing import Union

import numpy as np
from gym import spaces

from anvil_rl.common.type_aliases import Tensor
from anvil_rl.common.utils import get_space_shape, torch_to_numpy
from anvil_rl.models.actor_critics import Actor, ActorCritic
from anvil_rl.models.utils import get_mlp_size


class BaseExplorer(object):
    """
    Base explorer class that should be used when no extra noise is added to the actions/a stochastic actor is used

    :param actor: actor or actor critic network
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
        self.action_size = get_mlp_size(get_space_shape(action_space))

    def __call__(
        self, actor: Union[Actor, ActorCritic], observation: Tensor, step: int
    ) -> np.ndarray:
        if step >= self.start_steps:
            action = torch_to_numpy(actor(observation))
            if isinstance(self.action_space, spaces.Box):
                action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = self.action_space.sample()

        return action
