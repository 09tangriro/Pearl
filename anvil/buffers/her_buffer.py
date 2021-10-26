from typing import Dict, Union

import numpy as np
import torch as T
from gym.core import GoalEnv
from gym.spaces.space import Space

from anvil.buffers.base_buffer import BaseBuffer
from anvil.common.enumerations import GoalSelectionStrategy, TrajectoryType
from anvil.common.type_aliases import Trajectories


class HERBuffer(BaseBuffer):
    """Hindsight Exprience Replay (HER) Buffer"""

    def __init__(
        self,
        env: GoalEnv,
        buffer_size: int,
        observation_space: Space,
        action_space: Space,
        n_envs: int = 1,
        goal_selection_strategy: Union[str, GoalSelectionStrategy] = "future",
        n_sampled_goal: int = 4,
        device: Union[str, T.device] = "auto",
    ) -> None:
        super().__init__(
            buffer_size, observation_space, action_space, n_envs=n_envs, device=device
        )
        self.env = env
        self.desired_goals = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        self.next_achieved_goals = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )

        # Keep track of where in the data structure episodes end
        # and which transitions belong to which episodes.
        self.episode_end_indices = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.int8
        )
        self.index_episode_map = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.int8
        )
        self.episode = 0

        self._check_system_memory(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.desired_goals,
            self.next_achieved_goals,
            self.index_episode_map,
        )

        if isinstance(goal_selection_strategy, str):
            self.goal_section_strategy = GoalSelectionStrategy(
                goal_selection_strategy.lower()
            )
        else:
            self.goal_section_strategy = goal_selection_strategy

        self.her_ratio = 1 - (1.0 / (n_sampled_goal + 1))

    def _select_goal(self) -> np.ndarray:
        if self.goal_section_strategy == GoalSelectionStrategy.FINAL:
            return self.next_observations[self.pos]

    def add_trajectory(
        self,
        observation: Dict[str, np.ndarray],
        action: Union[np.ndarray, int],
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        self.observations[self.pos] = observation["observation"]
        self.desired_goals[self.pos] = observation["desired_goal"]
        self.rewards[self.pos] = reward
        self.actions[self.pos] = action
        self.observations[(self.pos + 1) % self.buffer_size] = next_observation[
            "observation"
        ]
        self.next_achieved_goals[self.pos] = next_observation["achieved_goal"]
        self.dones[self.pos] = done
        self.index_episode_map[self.pos] = self.episode
        self.pos += 1

        if done:
            self.episode_end_indices[self.episode] = self.pos
            self.episode += 1

            if self.episode == self.buffer_size:
                self.episode = 0

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _sample_goals(self, her_inds: np.ndarray) -> np.ndarray:
        her_episodes = self.index_episode_map[her_inds]
        episode_end_indices = self.episode_end_indices[her_episodes].reshape(
            her_episodes.shape
        )

        # Goal is the last state in the episode
        if self.goal_section_strategy == GoalSelectionStrategy.FINAL:
            goal_indices = episode_end_indices - 1

        # Goal is a random state in the same episode observed after current transition
        elif self.goal_section_strategy == GoalSelectionStrategy.FUTURE:
            # Need to expand her_inds to account for multiple environments
            her_inds = her_inds.reshape([-1, 1])
            her_inds = np.tile(her_inds, (1, self.n_envs))  # (batch_size, n_envs)
            # FAILURE MODE: if episode overlaps from end to beginning of buffer
            # then the randint method will likely fail due to low > high.
            # This will only happen at the overlapping episodes so quick
            # fix is to simply revert to final goal strategy in this case.
            if any(episode_end_indices < her_inds):
                goal_indices = episode_end_indices - 1
            else:
                goal_indices = np.random.randint(her_inds, episode_end_indices)

        else:
            raise ValueError(
                f"Strategy {self.goal_section_strategy} for samping goals not supported."
            )

        return self.next_achieved_goals[goal_indices].reshape(
            goal_indices.shape + self.obs_shape
        )

    def _sample_trajectories(
        self, batch_size: int, batch_inds: np.ndarray, dtype: Union[str, TrajectoryType]
    ) -> Trajectories:
        her_batch_size = int(batch_size * self.her_ratio)

        # Separate HER and replay batch indices
        her_inds = batch_inds[:her_batch_size]
        replay_inds = batch_inds[her_batch_size:]

        her_goals = self._sample_goals(her_inds)
        # the new state depends on the previous state and action
        # s_{t+1} = f(s_t, a_t)
        # so the next_achieved_goal depends also on the previous state and action
        # because we are in a GoalEnv:
        # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
        # therefore we have to use "next_achieved_goal" and not "achieved_goal"
        her_rewards = self.env.compute_reward(
            self.next_achieved_goals[her_inds], her_goals, {}
        ).reshape(len(her_goals), self.n_envs, 1)

        desired_goals = np.concatenate([her_goals, self.desired_goals[replay_inds]])
        rewards = np.concatenate([her_rewards, self.rewards[replay_inds]])
        observations = np.concatenate(
            [self.observations[batch_inds], desired_goals], axis=2
        )
        next_observations = np.concatenate(
            [self.observations[(batch_inds + 1) % self.buffer_size], desired_goals],
            axis=2,
        )
        actions = self.actions[batch_inds]
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

    def sample(
        self,
        batch_size: int,
        online: bool = False,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> Trajectories:
        if online:
            return self.last(batch_size, dtype)
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())

        # Sample transitions from the last complete episode recorded
        end_idx = self.episode_end_indices[self.episode - 1]
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + end_idx
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, end_idx, size=batch_size)

        return self._sample_trajectories(batch_size, batch_inds, dtype)

    def last(
        self, batch_size: int, dtype: Union[str, TrajectoryType] = "numpy"
    ) -> Trajectories:
        if isinstance(dtype, str):
            dtype = TrajectoryType(dtype.lower())
        assert batch_size < self.buffer_size

        # Sample transitions from the last complete episode recorded
        end_idx = self.episode_end_indices[self.episode - 1]
        start_idx = end_idx - batch_size
        if start_idx < 0 and self.full:
            batch_inds = np.concatenate((np.arange(start_idx, 0), np.arange(end_idx)))
        elif start_idx >= 0:
            batch_inds = np.arange(start_idx, end_idx)
        else:
            raise RuntimeError(
                f"Not enough samples collected, max batch_size={end_idx}"
            )

        return self._sample_trajectories(batch_size, batch_inds, dtype)
