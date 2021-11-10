from typing import Union

import numpy as np
import torch as T
from gym import Env

from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.enumerations import TrajectoryType
from anvilrl.common.type_aliases import Trajectories


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer handles sample collection and processing for on-policy algorithms.

    :param env: the environment
    :param buffer_size: max number of elements in the buffer
    """

    def __init__(
        self,
        env: Env,
        buffer_size: int,
        device: Union[str, T.device] = "auto",
    ) -> None:
        super().__init__(
            env,
            buffer_size,
            device,
        )
        self.next_observations = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=env.observation_space.dtype,
        )
        self._check_system_memory(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.next_observations,
        )

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=self.env.observation_space.dtype,
        )

    def add_trajectory(
        self,
        observation: np.ndarray,
        action: Union[np.ndarray, int],
        reward: Union[float, np.ndarray],
        next_observation: np.ndarray,
        done: Union[bool, np.ndarray],
    ) -> None:
        self.observations[self.pos] = observation
        self.next_observations[self.pos] = next_observation
        self.actions[self.pos] = np.array(action).reshape(*self.actions.shape[1:])
        self.rewards[self.pos] = np.array(reward).reshape(*self.rewards.shape[1:])
        self.dones[self.pos] = np.array(done).reshape(*self.dones.shape[1:])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> Trajectories:
        if self.full:
            assert (
                batch_size <= self.buffer_size
            ), f"Requesting {batch_size} samples when only {self.buffer_size} samples have been collected"
            upper_bound = self.buffer_size
        else:
            assert (
                batch_size <= self.pos
            ), f"Requesting {batch_size} samples when only {self.pos} samples have been collected"
            upper_bound = self.pos
        start_idx = np.random.randint(0, (upper_bound + 1) - batch_size)
        last_idx = start_idx + batch_size

        observations = self.observations[start_idx:last_idx]
        actions = self.actions[start_idx:last_idx]
        rewards = self.rewards[start_idx:last_idx]
        next_observations = self.next_observations[start_idx:last_idx]
        dones = self.dones[start_idx:last_idx]

        return self._transform_samples(
            flatten_env, dtype, observations, actions, rewards, next_observations, dones
        )

    def last(
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> Trajectories:
        assert batch_size <= self.buffer_size

        start_idx = self.pos - batch_size
        if start_idx < 0:
            batch_inds = np.concatenate((np.arange(start_idx, 0), np.arange(self.pos)))
        else:
            batch_inds = np.arange(start_idx, self.pos)

        observations = self.observations[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds]
        next_observations = self.next_observations[batch_inds]
        dones = self.dones[batch_inds]

        return self._transform_samples(
            flatten_env, dtype, observations, actions, rewards, next_observations, dones
        )

    def all(self) -> Trajectories:
        return Trajectories(
            observations=self.observations[: self.pos],
            actions=self.actions[: self.pos],
            rewards=self.rewards[: self.pos],
            next_observations=self.next_observations[: self.pos],
            dones=self.dones[: self.pos],
        )
