from typing import Union

import numpy as np
import torch as T
from gym import Env

from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.enumerations import TrajectoryType
from anvilrl.common.type_aliases import Trajectories


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
        self._check_system_memory(
            self.observations, self.actions, self.rewards, self.dones
        )

    def reset(self) -> None:
        super().reset()

    def add_trajectory(
        self,
        observation: np.ndarray,
        action: Union[np.ndarray, int],
        reward: Union[float, np.ndarray],
        next_observation: np.ndarray,
        done: Union[bool, np.ndarray],
    ) -> None:
        self.observations[self.pos] = observation
        self.observations[(self.pos + 1) % self.buffer_size] = next_observation
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
            flatten_env, dtype, observations, actions, rewards, next_observations, dones
        )

    def last(
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
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
            flatten_env, dtype, observations, actions, rewards, next_observations, dones
        )

    def all(self) -> Trajectories:
        return Trajectories(
            observations=self.observations[: self.pos],
            actions=self.actions[: self.pos],
            rewards=self.rewards[: self.pos],
            next_observations=self.observations[: (self.pos + 1) % self.buffer_size],
            dones=self.dones[: self.pos],
        )
