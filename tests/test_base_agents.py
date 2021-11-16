import shutil

import gym
import numpy as np
import torch as T

from anvilrl.agents.base_agents import BaseDeepAgent
from anvilrl.buffers import ReplayBuffer
from anvilrl.models.actor_critics import Actor, ActorCritic, Critic
from anvilrl.models.encoders import IdentityEncoder
from anvilrl.models.heads import ContinuousQHead
from anvilrl.models.torsos import MLP
from anvilrl.settings import ExplorerSettings, LoggerSettings


class MockDeepAgent(BaseDeepAgent):
    def _fit():
        pass


env = gym.make("Pendulum-v0")
envs = gym.vector.make("Pendulum-v0", num_envs=2, asynchronous=False)

encoder = IdentityEncoder()
torso = MLP(layer_sizes=[3, 64, 32], activation_fn=T.nn.ReLU)
head = ContinuousQHead(input_shape=32)
model = ActorCritic(
    actor=Actor(encoder, torso, head), critic=Critic(encoder, torso, head)
)

agent = MockDeepAgent(
    env=env,
    model=model,
    buffer_class=ReplayBuffer,
    explorer_settings=ExplorerSettings(start_steps=0),
    logger_settings=LoggerSettings(tensorboard_log_path="runs/tests"),
)
vec_agent = MockDeepAgent(
    env=envs,
    model=model,
    buffer_class=ReplayBuffer,
    explorer_settings=ExplorerSettings(start_steps=0),
    logger_settings=LoggerSettings(tensorboard_log_path="runs/tests"),
)
shutil.rmtree("runs/tests")


def test_step_env():
    env.seed(0)
    observation = env.reset()
    action = model(observation).detach().numpy()
    expected_next_obs, _, _, _ = env.step(action)
    actual_next_obs = agent.step_env(observation)

    assert isinstance(actual_next_obs, np.ndarray)

    observation = envs.reset()
    action = model(observation).detach().numpy()
    expected_next_obs, _, _, _ = env.step(action)
    actual_next_obs = vec_agent.step_env(observation)

    assert isinstance(actual_next_obs, np.ndarray)
