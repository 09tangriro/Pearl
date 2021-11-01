import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import psutil
import torch as T
from gym import Env

from anvil_rl.common.enumerations import TrajectoryType
from anvil_rl.common.type_aliases import Trajectories
from anvil_rl.common.utils import get_device, get_space_shape


class BaseBuffer(ABC):
    """
    the base buffer class which handles sample collection and processing.

    :param env: the environment
    :param buffer_size: max number of elements in the buffer
    :param n_envs: number of parallel environments
    """

    def __init__(
        self,
        env: Env,
        buffer_size: int,
        n_envs: int = 1,
        device: Union[str, T.device] = "auto",
    ) -> None:
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = get_device(device)
        self.full = False
        self.pos = 0

        # If only 1 environment, don't need the n_envs axis
        if n_envs > 1:
            self.batch_shape = (buffer_size, n_envs)
        else:
            self.batch_shape = (buffer_size,)

        self.obs_shape = get_space_shape(env.observation_space)
        action_shape = get_space_shape(env.action_space)

        self.observations = np.zeros(
            self.batch_shape + self.obs_shape,
            dtype=env.observation_space.dtype,
        )
        self.actions = np.zeros(
            self.batch_shape + action_shape,
            dtype=env.observation_space.dtype,
        )
        # Use 3 dims for easier calculations without having to think about broadcasting
        self.rewards = np.zeros(self.batch_shape + (1,), dtype=np.float32)
        self.dones = np.zeros(self.batch_shape + (1,), dtype=np.float32)

    @staticmethod
    def _check_system_memory(*buffers) -> None:
        """Check that the replay buffer can fit into memory"""
        mem_available = psutil.virtual_memory().available
        total_memory_usage = sum([buffer.nbytes for buffer in buffers])

        if total_memory_usage > mem_available:
            # Convert to GB
            total_memory_usage /= 1e9
            mem_available /= 1e9
            warnings.warn(
                "This system does not have enough memory to store the complete "
                f"replay buffer: {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    def _transform_samples(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        dtype: Union[str, TrajectoryType],
    ) -> Trajectories:
        """
        Handle post-processing of sampled trajectories.
        For now, only functionality is to transform to torch tensor if specified.
        """
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())

        # return torch tensors instead of numpy arrays
        if dtype == TrajectoryType.TORCH:
            observations = T.tensor(observations).to(self.device)
            actions = T.tensor(actions).to(self.device)
            rewards = T.tensor(rewards).to(self.device)
            next_observations = T.tensor(next_observations).to(self.device)
            dones = T.tensor(dones).to(self.device)

        return Trajectories(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )

    @abstractmethod
    def add_trajectory(
        self,
        observation: np.ndarray,
        action: Union[np.ndarray, int],
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a trajectory to the buffer

        :param observation: the observation
        :param action: the action taken
        :param reward: the reward received
        :param next_observation: the next observation collected
        :param done: the trajectory done flag
        """

    def add_batch_trajectories(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ):
        """
        Add a batch of trajectories to the buffer

        :param observations: the observations
        :param action: the actions taken
        :param reward: the rewards received
        :param next_observations: the next observations collected
        :param done: the trajectory done flags
        """
        for data in zip(observations, actions, rewards, next_observations, dones):
            self.add_trajectory(*data)

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

    @abstractmethod
    def last(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
    ) -> Trajectories:
        """
        Get the most recent batch of trajectories stored

        :param batch_size: the batch size
        :param dtype: whether to return the trajectories as "numpy" or "torch", default numpy
        :return: the trajectories
        """
