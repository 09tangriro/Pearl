from typing import Union

import numpy as np
import torch as T
from gym import Space

from anvil.buffers.base_buffer import BaseBuffer
from anvil.common.type_aliases import Trajectories, TrajectoryType


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
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        self._check_system_memory()

    def add_trajectory(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
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
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())
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