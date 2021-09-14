import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import psutil
import torch as T
from gym import Space

from anvil.buffers.utils import get_space_shape
from anvil.common.type_aliases import Trajectories, TrajectoryType
from anvil.common.utils import get_device


class BaseBuffer(ABC):
    """
    the base buffer class which handles sample collection and processing.

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
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.infinite_horizon = infinite_horizon
        self.device = get_device(device)
        self.full = False
        self.pos = 0

        obs_shape = get_space_shape(observation_space)
        action_shape = get_space_shape(action_space)

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + obs_shape, dtype=observation_space.dtype
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs) + action_shape,
            dtype=observation_space.dtype,
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self._check_system_memory()

    def _check_system_memory(self) -> None:
        """Check that the replay buffer can fit into memory"""
        mem_available = psutil.virtual_memory().available
        total_memory_usage = (
            self.observations.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
        )

        if total_memory_usage > mem_available:
            # Convert to GB
            total_memory_usage /= 1e9
            mem_available /= 1e9
            warnings.warn(
                "This system does not have enough memory to store the complete "
                f"replay buffer: {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    def add_trajectory(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        Add a trajectory to the buffer

        :param obs: the observation
        :param action: the action taken
        :param reward: the reward received
        :param next_obs: the next observation collected
        :param done: the trajectory done flag
        """
        self.observations[self.pos] = obs
        self.observations[(self.pos + 1) % self.buffer_size] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    @abstractmethod
    def sample(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
    ) -> Trajectories:
        """
        Sample a batch of trajectories

        :param batch_size: the batch size
        :param dtype: whether to return the trajectories as "numpy" or "torch", default numpy
        :return: the trajectories
        """
