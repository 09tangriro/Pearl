from typing import Union

import numpy as np
import torch as T
from gym import Env

from anvil_rl.buffers.base_buffer import BaseBuffer
from anvil_rl.common.enumerations import TrajectoryType
from anvil_rl.common.type_aliases import Trajectories


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer handles sample collection and processing for off-policy algorithms.
    We use a single array to save space in handling observations and next observations
    rather than using two different arrays. This assumes observations are stored sequentially
    and that observations can be 'blended' only if done = True since the next_observation
    in this case is essentially discarded anyway in most objective functions (at the end of
    the trajectory the value or q function of the terminal state is always 0!).

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
        self._check_system_memory(
            self.observations, self.actions, self.rewards, self.dones
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
        self.observations[(self.pos + 1) % self.buffer_size] = next_observation
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
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        observations = self.observations[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds]
        next_observations = self.observations[(batch_inds + 1) % self.buffer_size]
        dones = self.dones[batch_inds]

        return self._transform_samples(
            observations, actions, rewards, next_observations, dones, dtype
        )

    def last(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
    ) -> Trajectories:
        assert batch_size < self.buffer_size

        start_idx = self.pos - batch_size
        if start_idx < 0:
            batch_inds = np.concatenate((np.arange(start_idx, 0), np.arange(self.pos)))
        else:
            batch_inds = np.arange(start_idx, self.pos)

        observations = self.observations[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds]
        next_observations = self.observations[(batch_inds + 1) % self.buffer_size]
        dones = self.dones[batch_inds]

        return self._transform_samples(
            observations, actions, rewards, next_observations, dones, dtype
        )
