import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import psutil
import torch as T
from gym import Env
from gym.vector import VectorEnv

from anvilrl.common.enumerations import TrajectoryType
from anvilrl.common.type_aliases import Observation, Trajectories
from anvilrl.common.utils import get_device, get_space_shape


class BaseBuffer(ABC):
    """
    the base buffer class which handles sample collection and processing.

    :param env: the environment
    :param buffer_size: max number of elements in the buffer
    :param device: if return torch tensors on sampling, the device to attach to
    """

    def __init__(
        self,
        env: Env,
        buffer_size: int,
        device: Union[str, T.device] = "auto",
    ) -> None:
        self.env = env
        self.buffer_size = buffer_size
        self.device = get_device(device)
        self.full = False
        self.pos = 0

        self.num_envs = env.num_envs if isinstance(env, VectorEnv) else 1

        # If only 1 environment, don't need the num_envs axis
        if self.num_envs > 1:
            self.batch_shape = (buffer_size, self.num_envs)
        else:
            self.batch_shape = (buffer_size,)

        self.obs_shape = get_space_shape(env.observation_space)
        self.action_shape = get_space_shape(env.action_space)

        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=env.observation_space.dtype,
        )
        self.actions = np.zeros(
            (self.buffer_size,) + self.action_shape,
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

    def _flatten_env_axis(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Flatten the n_env axis of the input arrays to get samples of (batch_size, ...)

        :param data: the data to flatten (batch_size, num_envs, ...)
        :return: the flattened data (batch_size, ...)
        """
        if self.num_envs > 1:
            batch_size = data.shape[0] // self.num_envs
            data = data[-batch_size:].reshape(
                batch_size * self.num_envs, *data.shape[2:]
            )

        return data

    def _transform_samples(
        self,
        flatten_env: bool,
        dtype: Union[str, TrajectoryType],
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ) -> Trajectories:
        """
        Handle post-processing of sampled trajectories:
        1. Transform to torch tensor if specified
        2. Flatten the n_env axis if specified

        :param flatten_env: whether to flatten the num_envs axis
        :param dtype: the data type to return (torch or numpy)
        :param observations: the observations
        :param actions: the actions
        :param rewards: the rewards
        :param next_observations: the next observations
        :param dones: the done flags
        :return: the final transformed trajectories
        """
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())

        if flatten_env:
            observations = self._flatten_env_axis(observations)
            actions = self._flatten_env_axis(actions)
            rewards = self._flatten_env_axis(rewards)
            next_observations = self._flatten_env_axis(next_observations)
            dones = self._flatten_env_axis(dones)

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
    def reset(self) -> None:
        """Reset the buffer"""

        self.pos = 0
        self.full = False

        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=self.env.observation_space.dtype,
        )
        self.actions = np.zeros(
            (self.buffer_size,) + self.action_shape,
            dtype=self.env.observation_space.dtype,
        )
        self.rewards = np.zeros(self.batch_shape + (1,), dtype=np.float32)
        self.dones = np.zeros(self.batch_shape + (1,), dtype=np.float32)

    @abstractmethod
    def add_trajectory(
        self,
        observation: Observation,
        action: Union[np.ndarray, int],
        reward: Union[float, np.ndarray],
        next_observation: Observation,
        done: Union[bool, np.ndarray],
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
        observations: Observation,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: Observation,
        dones: np.ndarray,
    ) -> None:
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
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> Trajectories:
        """
        Sample a batch of trajectories

        :param batch_size: the batch size
        :param flatten_env: useful for multiple environments, whether to sample with the num_envs axis
        :param dtype: whether to return the trajectories as "numpy" or "torch", default numpy
        :return: the sampled trajectories
        """

    @abstractmethod
    def last(
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> Trajectories:
        """
        Get the most recent batch of trajectories stored

        :param batch_size: the batch size
        :param flatten_env: useful for multiple environments, whether to sample with the num_envs axis
        :param dtype: whether to return the trajectories as "numpy" or "torch", default numpy
        :return: the most recent trajectories
        """

    @abstractmethod
    def all(self) -> Trajectories:
        """
        Get all stored trajectories

        :return: stored trajectories
        """
