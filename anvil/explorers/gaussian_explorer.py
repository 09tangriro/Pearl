from typing import Any, Optional, Union

import torch as T
from gym import spaces

from anvil.common.type_aliases import Tensor
from anvil.explorers.base_explorer import BaseExplorer
from anvil.models.actor_critics import Actor, ActorCritic


class GaussianExplorer(BaseExplorer):
    def __init__(
        self,
        actor: Union[Actor, ActorCritic],
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
        super().__init__(actor, action_space, start_steps=start_steps)
        self.scale = scale

    def __call__(self, observation: Tensor, step: int) -> T.Tensor:
        actions = super().__call__(observation, step)
        if step > self.start_steps:
            noises = T.normal(mean=0.0, std=self.scale, size=self.action_space.shape)
            actions = actions + noises
            actions = T.clip(actions, self.low, self.high)
        return actions
