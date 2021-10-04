from typing import Union

import numpy as np
import torch as T
from gym import Space

from anvil.buffers.base_buffer import BaseBuffer
from anvil.common.enumerations import TrajectoryType
from anvil.common.type_aliases import Trajectories


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer handles sample collection and processing for off-policy algorithms.
    We use a single array to save space in handling observations and next observations
    rather than using two different arrays. This assumes observations are stored sequentially
    and that observations can be 'blended' only if done = True since the next_observation
    in this case is essentially discarded anyway in most objective functions (at the end of
    the trajectory the value or q function of the terminal state is always 0!).

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
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
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
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())
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

    def last(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
    ) -> Trajectories:
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())
        assert batch_size <= self.buffer_size - 1

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
