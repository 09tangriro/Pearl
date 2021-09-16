import warnings
from typing import Union

import numpy as np
import psutil
import torch as T
from gym import Space

from anvil.buffers.base_buffer import BaseBuffer
from anvil.buffers.utils import get_space_shape
from anvil.common.type_aliases import Trajectories, TrajectoryType
from anvil.common.utils import get_device


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer handles sample collection and processing for off-policy algorithms.

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
        self._check_system_memory()

    def add_trajectory(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.observations[self.pos] = obs
        self.observations[(self.pos + 1) % self.buffer_size] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
    ) -> Trajectories:
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        obs = self.observations[batch_inds, 0, :]
        actions = self.actions[batch_inds, 0, :]
        rewards = self.rewards[batch_inds]
        next_obs = self.observations[(batch_inds + 1) % self.buffer_size, 0, :]
        dones = self.dones[batch_inds]

        # return torch tensors instead of numpy arrays
        if dtype == TrajectoryType.TORCH:
            obs = T.tensor(obs).to(self.device)
            actions = T.tensor(actions).to(self.device)
            rewards = T.tensor(rewards).to(self.device)
            next_obs = T.tensor(next_obs).to(self.device)
            dones = T.tensor(dones).to(self.device)

        return Trajectories(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            dones=dones,
        )
