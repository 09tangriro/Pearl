from argparse import ArgumentParser
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import gym
import numpy as np
import torch as T

from anvilrl.agents.a2c import A2C
from anvilrl.agents.ddpg import DDPG
from anvilrl.agents.dqn import DQN
from anvilrl.agents.es import ES
from anvilrl.agents.ga import GA
from anvilrl.buffers import HERBuffer
from anvilrl.common.utils import get_space_shape
from anvilrl.models.actor_critics import (
    ActorCriticWithCriticTarget,
    Critic,
    DeepIndividual,
    EpsilonGreedyActor,
    Individual,
)
from anvilrl.models.encoders import DictEncoder, IdentityEncoder
from anvilrl.models.heads import DiagGaussianHead, DiscreteQHead
from anvilrl.models.torsos import MLP
from anvilrl.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    PopulationInitializerSettings,
)


def dqn_demo():
    env = gym.make("CartPole-v0")
    agent = DQN(
        env=env,
        model=None,
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/DQN-demo", verbose=True
        ),
        explorer_settings=ExplorerSettings(start_steps=1000),
    )
    agent.fit(
        num_steps=50000, batch_size=32, critic_epochs=16, train_frequency=("episode", 1)
    )


def dqn_parallel_demo():
    env = gym.vector.make("CartPole-v0", 10, asynchronous=False)
    encoder = IdentityEncoder()
    torso = MLP(layer_sizes=[4, 64, 32], activation_fn=T.nn.ReLU)
    head = DiscreteQHead(input_shape=32, output_shape=2)

    actor = EpsilonGreedyActor(
        critic_encoder=encoder, critic_torso=torso, critic_head=head
    )
    critic = Critic(encoder=encoder, torso=torso, head=head)
    model = ActorCriticWithCriticTarget(actor=actor, critic=critic)

    agent = DQN(
        env=env,
        model=model,
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/DQN-parallel-demo", verbose=True
        ),
        explorer_settings=ExplorerSettings(start_steps=1000),
    )
    agent.fit(
        num_steps=50000,
        batch_size=320,
        critic_epochs=16,
        train_frequency=("episode", 1),
    )


def ddpg_demo():
    env = gym.make("Pendulum-v0")
    agent = DDPG(
        env=env,
        model=None,
        logger_settings=LoggerSettings(tensorboard_log_path="runs/DDPG-demo"),
        explorer_settings=ExplorerSettings(start_steps=1000, scale=0.1),
    )
    agent.fit(
        num_steps=100000,
        batch_size=64,
        critic_epochs=1,
        actor_epochs=1,
        train_frequency=("step", 1),
    )


def es_demo():
    class Sphere(gym.Env):
        """
        Sphere(2) function for testing ES agent.
        """

        def __init__(self):
            self.action_space = gym.spaces.Box(low=-100, high=100, shape=(2,))
            self.observation_space = gym.spaces.Discrete(1)

        def step(self, action):
            return 0, -(action[0] ** 2 + action[1] ** 2), False, {}

        def reset(self):
            return 0

    POPULATION_SIZE = 10
    env = gym.vector.SyncVectorEnv([lambda: Sphere() for _ in range(POPULATION_SIZE)])
    model = Individual(env.single_action_space, np.array([10, 10]))

    agent = ES(
        env=env,
        model=model,
        learning_rate=1,
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/ES-demo", log_frequency=("step", 1)
        ),
    )
    agent.fit(num_steps=20)


def es_deep_demo():
    POPULATION_SIZE = 100
    env = gym.vector.make("Pendulum-v0", POPULATION_SIZE, asynchronous=False)
    observation_shape = get_space_shape(env.single_observation_space)
    encoder = IdentityEncoder()
    torso = MLP(layer_sizes=[observation_shape[0], 64, 32], activation_fn=T.nn.ReLU)
    head = DiagGaussianHead(input_shape=32, action_size=1)
    model = DeepIndividual(encoder=encoder, torso=torso, head=head)

    agent = ES(
        env=env,
        model=model,
        learning_rate=0.1,
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/DeepES-demo", log_frequency=("episode", 1)
        ),
    )
    agent.fit(num_steps=50000, epochs=8, train_frequency=("step", 50))


def ga_demo():
    class Mastermind(gym.Env):
        """
        Mastermind game for testing GA agent.
        """

        def __init__(self):
            self.action_space = gym.spaces.MultiDiscrete([4, 4, 4, 4])
            self.observation_space = gym.spaces.Discrete(1)
            self.master = [0, 1, 2, 3]

        def step(self, action):
            p1 = 0
            p2 = 0
            p1_map = [0] * 4
            p2_map = [0] * 4
            Mp2_map = [0] * 4

            for i, colour in enumerate(action):
                if colour == self.master[i]:
                    p1_map[i] = 1
                for el in self.master:
                    if colour == el:
                        p2_map[i] = 1
                for i, el in enumerate(self.master):
                    for colour in action:
                        if colour == el:
                            Mp2_map[i] = 1

            Mp2_map = [m - n for m, n in zip(Mp2_map, p1_map)]
            p2_map = [m - n for m, n in zip(p2_map, p1_map)]

            p1 = sum(p1_map)
            p2 = min(sum(p2_map), sum(Mp2_map))

            return 0, p1 + 0.5 * p2, False, {}

        def reset(self):
            return 0

    POPULATION_SIZE = 10
    env = gym.vector.SyncVectorEnv(
        [lambda: Mastermind() for _ in range(POPULATION_SIZE)]
    )
    model = Individual(env.single_action_space)

    agent = GA(
        env=env,
        model=model,
        population_settings=PopulationInitializerSettings(strategy="uniform"),
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/GA-demo", log_frequency=("step", 1)
        ),
    )
    agent.fit(num_steps=25)


def her_demo():
    class BitFlippingEnv(gym.GoalEnv):
        """
        Simple bit flipping env, useful to test HER.
        The goal is to flip all the bits to get a vector of ones.
        In the continuous variant, if the ith action component has a value > 0,
        then the ith bit will be flipped.
        :param n_bits: Number of bits to flip
        :param continuous: Whether to use the continuous actions version or not,
            by default, it uses the discrete one
        :param max_steps: Max number of steps, by default, equal to n_bits
        :param discrete_obs_space: Whether to use the discrete observation
            version or not, by default, it uses the ``MultiBinary`` one
        :param image_obs_space: Use image as input instead of the ``MultiBinary`` one.
        :param channel_first: Whether to use channel-first or last image.
        """

        spec = gym.envs.registration.EnvSpec("BitFlippingEnv-v0")

        def __init__(
            self,
            n_bits: int = 10,
            continuous: bool = False,
            max_steps: Optional[int] = None,
            discrete_obs_space: bool = False,
            image_obs_space: bool = False,
            channel_first: bool = True,
        ):
            super(BitFlippingEnv, self).__init__()
            # Shape of the observation when using image space
            self.image_shape = (1, 36, 36) if channel_first else (36, 36, 1)
            # The achieved goal is determined by the current state
            # here, it is a special where they are equal
            if discrete_obs_space:
                # In the discrete case, the agent act on the binary
                # representation of the observation
                self.observation_space = gym.spaces.Dict(
                    {
                        "observation": gym.spaces.Discrete(2 ** n_bits),
                        "achieved_goal": gym.spaces.Discrete(2 ** n_bits),
                        "desired_goal": gym.spaces.Discrete(2 ** n_bits),
                    }
                )
            elif image_obs_space:
                # When using image as input,
                # one image contains the bits 0 -> 0, 1 -> 255
                # and the rest is filled with zeros
                self.observation_space = gym.spaces.Dict(
                    {
                        "observation": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=self.image_shape,
                            dtype=np.uint8,
                        ),
                        "achieved_goal": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=self.image_shape,
                            dtype=np.uint8,
                        ),
                        "desired_goal": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=self.image_shape,
                            dtype=np.uint8,
                        ),
                    }
                )
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        "observation": gym.spaces.MultiBinary(n_bits),
                        "achieved_goal": gym.spaces.MultiBinary(n_bits),
                        "desired_goal": gym.spaces.MultiBinary(n_bits),
                    }
                )

            self.obs_space = gym.spaces.MultiBinary(n_bits)

            if continuous:
                self.action_space = gym.spaces.Box(
                    -1, 1, shape=(n_bits,), dtype=np.float32
                )
            else:
                self.action_space = gym.spaces.Discrete(n_bits)
            self.continuous = continuous
            self.discrete_obs_space = discrete_obs_space
            self.image_obs_space = image_obs_space
            self.state = None
            self.desired_goal = np.ones((n_bits,))
            if max_steps is None:
                max_steps = n_bits
            self.max_steps = max_steps
            self.current_step = 0

        def seed(self, seed: int) -> None:
            self.obs_space.seed(seed)

        def convert_if_needed(self, state: np.ndarray) -> Union[int, np.ndarray]:
            """
            Convert to discrete space if needed.
            :param state:
            :return:
            """
            if self.discrete_obs_space:
                # The internal state is the binary representation of the
                # observed one
                return int(sum([state[i] * 2 ** i for i in range(len(state))]))

            if self.image_obs_space:
                size = np.prod(self.image_shape)
                image = np.concatenate(
                    (state * 255, np.zeros(size - len(state), dtype=np.uint8))
                )
                return image.reshape(self.image_shape).astype(np.uint8)
            return state

        def convert_to_bit_vector(
            self, state: Union[int, np.ndarray], batch_size: int
        ) -> np.ndarray:
            """
            Convert to bit vector if needed.
            :param state:
            :param batch_size:
            :return:
            """
            # Convert back to bit vector
            if isinstance(state, int):
                state = np.array(state).reshape(batch_size, -1)
                # Convert to binary representation
                state = (
                    ((state[:, :] & (1 << np.arange(len(self.state))))) > 0
                ).astype(int)
            elif self.image_obs_space:
                state = state.reshape(batch_size, -1)[:, : len(self.state)] / 255
            else:
                state = np.array(state).reshape(batch_size, -1)

            return state

        def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
            """
            Helper to create the observation.
            :return: The current observation.
            """
            return OrderedDict(
                [
                    ("observation", self.convert_if_needed(self.state.copy())),
                    ("achieved_goal", self.convert_if_needed(self.state.copy())),
                    ("desired_goal", self.convert_if_needed(self.desired_goal.copy())),
                ]
            )

        def reset(self) -> Dict[str, Union[int, np.ndarray]]:
            self.current_step = 0
            self.state = self.obs_space.sample()
            return self._get_obs()

        def step(self, action: Union[np.ndarray, int]) -> Tuple:
            if self.continuous:
                self.state[action > 0] = 1 - self.state[action > 0]
            else:
                self.state[action] = 1 - self.state[action]
            obs = self._get_obs()
            reward = float(
                self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None)
            )
            done = reward == 0
            self.current_step += 1
            # Episode terminate when we reached the goal or the max number of steps
            info = {"is_success": done}
            done = done or self.current_step >= self.max_steps
            return obs, reward, done, info

        def compute_reward(
            self,
            achieved_goal: Union[int, np.ndarray],
            desired_goal: Union[int, np.ndarray],
            _info: Optional[Dict[str, Any]],
        ) -> np.float32:
            # As we are using a vectorized version, we need to keep track of the `batch_size`
            if isinstance(achieved_goal, int):
                batch_size = 1
            elif self.image_obs_space:
                batch_size = (
                    achieved_goal.shape[0] if len(achieved_goal.shape) > 3 else 1
                )
            else:
                batch_size = (
                    achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
                )

            desired_goal = self.convert_to_bit_vector(desired_goal, batch_size)
            achieved_goal = self.convert_to_bit_vector(achieved_goal, batch_size)

            # Deceptive reward: it is positive only when the goal is achieved
            # Here we are using a vectorized version
            distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(distance > 0).astype(np.float32)

        def render(self, mode: str = "human") -> Optional[np.ndarray]:
            if mode == "rgb_array":
                return self.state.copy()
            print(self.state)

        def close(self) -> None:
            pass

    env = BitFlippingEnv()

    action_size = env.action_space.n
    observation_size = 20

    encoder = DictEncoder()
    torso = MLP(layer_sizes=[observation_size, 256], activation_fn=T.nn.ReLU)
    head = DiscreteQHead(input_shape=256, output_shape=action_size)

    actor = EpsilonGreedyActor(
        critic_encoder=encoder, critic_torso=torso, critic_head=head
    )
    critic = Critic(encoder=encoder, torso=torso, head=head)
    model = ActorCriticWithCriticTarget(actor=actor, critic=critic)

    agent = DQN(
        env=env,
        model=model,
        buffer_class=HERBuffer,
        buffer_settings=BufferSettings(),
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/HER-demo", verbose=True
        ),
        explorer_settings=ExplorerSettings(start_steps=0),
    )

    agent.fit(
        num_steps=50000,
        batch_size=128,
        critic_epochs=40,
        train_frequency=("episode", 16),
    )


def a2c_demo():
    env = gym.make("CartPole-v0")
    batch_size = 5

    agent = A2C(
        env=env,
        buffer_settings=BufferSettings(buffer_size=batch_size),
        logger_settings=LoggerSettings(
            tensorboard_log_path="runs/A2C-demo", verbose=True
        ),
        entropy_coefficient=0,
    )

    agent.fit(
        num_steps=100000,
        batch_size=batch_size,
        critic_epochs=1,
        train_frequency=("step", batch_size),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="AnvilRL demo with preloaded hyperparameters")
    parser.add_argument("--agent", help="Agent to demo")
    kwargs = parser.parse_args()

    if kwargs.agent.lower() == "dqn":
        dqn_demo()
    elif kwargs.agent.lower() == "dqn-parallel":
        dqn_parallel_demo()
    elif kwargs.agent.lower() == "ddpg":
        ddpg_demo()
    elif kwargs.agent.lower() == "es":
        es_demo()
    elif kwargs.agent.lower() == "es-deep":
        es_deep_demo()
    elif kwargs.agent.lower() == "ga":
        ga_demo()
    elif kwargs.agent.lower() == "her":
        her_demo()
    elif kwargs.agent.lower() == "a2c":
        a2c_demo()
    else:
        raise ValueError(f"Agent {kwargs.agent} not found")
