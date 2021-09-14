from typing import Union

import numpy as np
import torch as T
from gym import Space

from anvil.buffers.base_buffer import BaseBuffer


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer handles sample collection and processing for on-policy algorithms.

    :param buffer_size: max number of elements in the buffer
    :param observation_space: observation space
    :param action_space: action space
    :param n_envs: number of parallel environments
    :param infinite_horizon: whether environment is episodic or not
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Space,
        action_space: Space,
        n_envs: int = 1,
        infinite_horizon: bool = False,
        device: Union[str, T.device] = "auto",
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            n_envs,
            infinite_horizon,
            device,
        )
