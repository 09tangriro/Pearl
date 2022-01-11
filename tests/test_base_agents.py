import shutil

import gym
import numpy as np
import torch as T

from pearll.agents.base_agents import BaseAgent
from pearll.buffers import ReplayBuffer
from pearll.common.type_aliases import Log
from pearll.common.utils import set_seed
from pearll.models.actor_critics import Actor, ActorCritic, Critic
from pearll.models.encoders import IdentityEncoder
from pearll.models.heads import ContinuousQHead
from pearll.models.torsos import MLP
from pearll.settings import ExplorerSettings, LoggerSettings


class MockRLAgent(BaseAgent):
    def _fit(self, batch_size, actor_epochs=1, critic_epochs=1):
        return Log(actor_loss=0, critic_loss=0, entropy=0, divergence=0)


env = gym.make("Pendulum-v0")
envs = gym.vector.make("Pendulum-v0", num_envs=2, asynchronous=False)

encoder = IdentityEncoder()
torso = MLP(layer_sizes=[3, 64, 32], activation_fn=T.nn.ReLU)
head = ContinuousQHead(input_shape=32)
model = ActorCritic(
    actor=Actor(encoder, torso, head), critic=Critic(encoder, torso, head)
)

deep_agent = MockRLAgent(
    env=env,
    model=model,
    buffer_class=ReplayBuffer,
    explorer_settings=ExplorerSettings(start_steps=0),
    logger_settings=LoggerSettings(tensorboard_log_path="runs/tests"),
)
vec_deep_agent = MockRLAgent(
    env=envs,
    model=model,
    buffer_class=ReplayBuffer,
    explorer_settings=ExplorerSettings(start_steps=0),
    logger_settings=LoggerSettings(tensorboard_log_path="runs/tests"),
)


def test_deep_step_env():
    set_seed(0, env)
    observation = env.reset()
    action = model(observation).detach().numpy()
    expected_next_obs, _, _, _ = env.step(action)
    set_seed(0, env)
    observation = env.reset()
    actual_next_obs = deep_agent.step_env(observation)
    np.testing.assert_array_equal(actual_next_obs, expected_next_obs)

    set_seed(0, env)
    observation = env.reset()
    final_episode = deep_agent.episode + 1
    for i in range(200):
        observation = deep_agent.step_env(observation)
        if i == 199:
            assert deep_agent.episode == final_episode
        else:
            assert deep_agent.episode != final_episode

    set_seed(0, envs)
    observation = envs.reset()
    action = model(observation).detach().numpy()
    expected_next_obs, _, _, _ = envs.step(action)
    set_seed(0, envs)
    observation = envs.reset()
    actual_next_obs = vec_deep_agent.step_env(observation)
    np.testing.assert_array_equal(actual_next_obs, expected_next_obs)


def test_deep_fit():
    deep_agent.step = 0
    deep_agent.episode = 0
    deep_agent.fit(num_steps=2, batch_size=1, train_frequency=("step", 1))
    assert deep_agent.episode == 0
    assert deep_agent.step == 2

    deep_agent.step = 0
    deep_agent.episode = 0
    deep_agent.fit(num_steps=200, batch_size=1, train_frequency=("episode", 1))
    assert deep_agent.step == 200
    assert deep_agent.episode == 1

    vec_deep_agent.step = 0
    vec_deep_agent.episode = 0
    vec_deep_agent.fit(num_steps=2, batch_size=1, train_frequency=("step", 1))
    assert vec_deep_agent.episode == 0
    assert vec_deep_agent.step == 2

    vec_deep_agent.step = 0
    vec_deep_agent.episode = 0
    vec_deep_agent.fit(num_steps=200, batch_size=1, train_frequency=("episode", 1))
    assert deep_agent.step == 200
    assert vec_deep_agent.episode == 1


shutil.rmtree("runs/tests")
