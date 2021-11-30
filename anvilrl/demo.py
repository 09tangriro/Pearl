from argparse import ArgumentParser

import gym
import numpy as np

from anvilrl.agents.ddpg import DDPG
from anvilrl.agents.dqn import DQN
from anvilrl.agents.es import ES
from anvilrl.agents.ga import GA
from anvilrl.settings import (
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

    agent = ES(
        env=env,
        population_init_settings=PopulationInitializerSettings(
            starting_point=np.array([10, 10])
        ),
        learning_rate=1,
        logger_settings=LoggerSettings(tensorboard_log_path="runs/ES-demo"),
    )
    agent.fit(num_steps=15)


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

    agent = GA(
        env=env,
        population_init_settings=PopulationInitializerSettings(strategy="uniform"),
        logger_settings=LoggerSettings(tensorboard_log_path="runs/GA-demo"),
    )
    agent.fit(num_steps=25)


if __name__ == "__main__":
    parser = ArgumentParser(description="AnvilRL demo with preloaded hyperparameters")
    parser.add_argument("--agent", help="Agent to demo")
    kwargs = parser.parse_args()

    if kwargs.agent.lower() == "dqn":
        dqn_demo()
    elif kwargs.agent.lower() == "ddpg":
        ddpg_demo()
    elif kwargs.agent.lower() == "es":
        es_demo()
    elif kwargs.agent.lower() == "ga":
        ga_demo()
    else:
        raise ValueError(f"Agent {kwargs.agent} not found")
