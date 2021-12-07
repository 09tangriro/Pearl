from typing import Dict, Tuple, Union

import numpy as np
import torch as T
from gym.core import GoalEnv

from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.common.enumerations import GoalSelectionStrategy, TrajectoryType
from anvilrl.common.type_aliases import DictTrajectories, Tensor


class HERBuffer(BaseBuffer):
    """
    Hindsight Exprience Replay (HER) Buffer
    Paper: https://arxiv.org/abs/1707.01495
    Guide: https://towardsdatascience.com/hindsight-experience-replay-her-implementation-92eebab6f653
    The HER buffer uses the same trick as the standard `ReplayBuffer` using
    a single array to handle the observations. Instead of sampling and storing
    goals every time we add transitions, instead we do it all at once when
    sampling for vectorized (fast) processing.

    TODO: DOES NOT YET SUPPORT MULTIPLE ENVIRONEMTS!! More testing needed for this.

    :param env: the environment
    :param buffer_size: max number of elements in the buffer
    :param goal_selection_strategy: the goal selection strategy to be used, defaults to future
    :param n_sampled_goal: ratio of HER data to data coming from normal experience replay
    :param device: if return torch tensors on sampling, the device to attach to
    """

    def __init__(
        self,
        env: GoalEnv,
        buffer_size: int,
        goal_selection_strategy: Union[str, GoalSelectionStrategy] = "future",
        n_sampled_goal: int = 4,
        device: Union[str, T.device] = "auto",
    ) -> None:
        super().__init__(env, buffer_size, device=device)
        self.env = env
        self.desired_goals = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=env.observation_space.dtype,
        )
        self.next_achieved_goals = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=env.observation_space.dtype,
        )

        # Keep track of where in the data structure episodes end
        self.episode_end_indices = np.zeros(self.batch_shape, dtype=np.uint32)
        # Keep track of which transitions belong to which episodes.
        self.index_episode_map = np.zeros(self.batch_shape, dtype=np.uint32)
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

    def reset(self) -> None:
        super().reset()
        self.episode = 0

        self.desired_goals = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=self.env.observation_space.dtype,
        )
        self.next_achieved_goals = np.zeros(
            (self.buffer_size,) + self.obs_shape,
            dtype=self.env.observation_space.dtype,
        )

        # Keep track of where in the data structure episodes end
        self.episode_end_indices = np.zeros(self.batch_shape, dtype=np.int8)
        # Keep track of which transitions belong to which episodes.
        self.index_episode_map = np.zeros(self.batch_shape, dtype=np.int8)

    def add_trajectory(
        self,
        observation: Dict[str, np.ndarray],
        action: Union[np.ndarray, int],
        reward: Union[float, np.ndarray],
        next_observation: Dict[str, np.ndarray],
        done: Union[bool, np.ndarray],
    ) -> None:
        self.observations[self.pos] = observation["observation"]
        self.desired_goals[self.pos] = observation["desired_goal"]
        self.rewards[self.pos] = np.array(reward).reshape(*self.rewards.shape[1:])
        self.actions[self.pos] = np.array(action).reshape(*self.actions.shape[1:])
        self.observations[(self.pos + 1) % self.buffer_size] = next_observation[
            "observation"
        ]
        self.next_achieved_goals[self.pos] = next_observation["achieved_goal"]
        self.dones[self.pos] = np.array(done).reshape(*self.dones.shape[1:])
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
        """
        Sample new episode goals to calculate rewards from

        :param her_inds: the batch indices designated for new goal sampling
        :return: the new episode goals
        """
        her_episodes = self.index_episode_map[her_inds]
        # Reshaping to handle multiple environments (if self.n_envs > 1)
        episode_end_indices = self.episode_end_indices[her_episodes].reshape(
            her_episodes.shape
        )

        # Goal is the last state in the episode
        if self.goal_section_strategy == GoalSelectionStrategy.FINAL:
            goal_indices = episode_end_indices - 1

        # Goal is a random state in the same episode observed after current transition
        elif self.goal_section_strategy == GoalSelectionStrategy.FUTURE:
            # FAILURE MODE: if episode overlaps from end to beginning of buffer
            # then the randint method will likely fail due to low > high.
            # This will only happen at the overlapping episodes so quick
            # fix is to simply revert to final goal strategy in this case.
            # TODO: This will also break with multiple environments, need some sort
            # of `np.tile()` to account for this.
            if any(episode_end_indices < her_inds):
                goal_indices = episode_end_indices - 1
            else:
                goal_indices = np.random.randint(her_inds, episode_end_indices)

        else:
            raise ValueError(
                f"Strategy {self.goal_section_strategy} for samping goals not supported."
            )

        # Reshaping to handle multiple environments (if self.n_envs > 1)
        return self.next_achieved_goals[goal_indices].reshape(
            goal_indices.shape + self.obs_shape
        )

    def _sample_trajectories(
        self,
        batch_inds: np.ndarray,
    ) -> Tuple[Dict, Tensor, Tensor, Dict, Tensor]:
        """
        Get the trajectories based on batch indices calculated

        :param batch_size: number of elements to sample, included to reduce processing instead of using `len(batch_inds)`
        :param batch_inds: the indices of the elements to sample
        :param dtype: :param dtype: whether to return the trajectories as "numpy" or "torch", default numpy
        """
        her_batch_size = int(len(batch_inds) * self.her_ratio)

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
        )[:, np.newaxis]

        desired_goals = np.concatenate([her_goals, self.desired_goals[replay_inds]])
        rewards = np.concatenate([her_rewards, self.rewards[replay_inds]])
        observations = {
            "observation": self.observations[batch_inds],
            "desired_goal": desired_goals,
        }
        next_observations = {
            "observation": self.observations[(batch_inds + 1) % self.buffer_size],
            "desired_goal": desired_goals,
        }
        actions = self.actions[batch_inds]
        dones = self.dones[batch_inds]

        return observations, actions, rewards, next_observations, dones

    def sample(
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> DictTrajectories:
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

        trajectories = self._sample_trajectories(batch_inds)
        return DictTrajectories(
            observations=trajectories[0],
            actions=trajectories[1],
            rewards=trajectories[2],
            next_observations=trajectories[3],
            dones=trajectories[4],
        )

    def last(
        self,
        batch_size: int,
        flatten_env: bool = True,
        dtype: Union[str, TrajectoryType] = "numpy",
    ) -> DictTrajectories:
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

        trajectories = self._sample_trajectories(batch_inds)
        return DictTrajectories(
            observations=trajectories[0],
            actions=trajectories[1],
            rewards=trajectories[2],
            next_observations=trajectories[3],
            dones=trajectories[4],
        )

    def all(self) -> DictTrajectories:
        return DictTrajectories(
            observations={
                "observation": self.observations[: self.pos],
                "desired_goal": self.desired_goals[: self.pos],
            },
            actions=self.actions[: self.pos],
            rewards=self.rewards[: self.pos],
            next_observations={
                "observation": self.observations[: (self.pos + 1) % self.buffer_size],
                "desired_goal": self.desired_goals[: (self.pos + 1) % self.buffer_size],
            },
            dones=self.dones[: self.pos],
        )
