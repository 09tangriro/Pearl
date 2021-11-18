from argparse import ArgumentParser

import gym
import numpy as np

from anvilrl.agents.ddpg import DDPG
from anvilrl.agents.dqn import DQN
from anvilrl.agents.es import ES
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
    else:
        raise ValueError(f"Agent {kwargs.agent} not found")
