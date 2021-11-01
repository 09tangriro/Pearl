from typing import Union

import numpy as np
import torch as T
from gym import Env

from anvil_rl.buffers.base_buffer import BaseBuffer
from anvil_rl.common.enumerations import TrajectoryType
from anvil_rl.common.type_aliases import Trajectories


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer handles sample collection and processing for on-policy algorithms.

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
        super().__init__(
            env,
            buffer_size,
            n_envs,
            device,
        )
        self.next_observations = np.zeros(
            self.batch_shape + self.obs_shape,
            dtype=env.observation_space.dtype,
        )
        self._check_system_memory(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.next_observations,
        )

    def add_trajectory(
        self,
        observation: np.ndarray,
        action: Union[np.ndarray, int],
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.pos] = observation
        self.rewards[self.pos] = reward
        self.actions[self.pos] = action
        self.next_observations[self.pos] = next_observation
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
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
            observations, actions, rewards, next_observations, dones, dtype
        )

    def last(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
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
            observations, actions, rewards, next_observations, dones, dtype
        )
