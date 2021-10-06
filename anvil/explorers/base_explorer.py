from typing import Any, Optional, Union

import numpy as np
import torch as T
from gym import spaces

from anvil.common.type_aliases import Tensor
from anvil.common.utils import get_space_shape, numpy_to_torch
from anvil.models.actor_critics import Actor, ActorCritic
from anvil.models.utils import get_mlp_size


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
        action_space: spaces.Space,
        start_steps: int = 20000,
    ) -> None:
        self.start_steps = start_steps
        self.actor = actor
        self.action_space = action_space
        self.action_size = get_mlp_size(get_space_shape(action_space))
        self.low = numpy_to_torch(action_space.low)
        self.high = numpy_to_torch(action_space.high)

    def __call__(self, observation: Tensor, step: int) -> T.Tensor:
        if step >= self.start_steps:
            action = self.actor(observation)
            if isinstance(self.action_space, spaces.Box):
                action = T.clip(action, self.low.item(), self.high.item())
        else:
            action = numpy_to_torch(
                np.random.uniform(
                    low=self.action_space.low,
                    high=self.action_space.high,
                    size=self.action_size,
                )
            )

        return action
