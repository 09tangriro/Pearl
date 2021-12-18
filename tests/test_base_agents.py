import shutil

import gym
import numpy as np
import torch as T

from anvilrl.agents.base_agents import BaseDeepAgent, BaseEvolutionAgent
from anvilrl.buffers import ReplayBuffer
from anvilrl.models.actor_critics import Actor, ActorCritic, Critic, Individual
from anvilrl.models.encoders import IdentityEncoder
from anvilrl.models.heads import ContinuousQHead
from anvilrl.models.torsos import MLP
from anvilrl.settings import ExplorerSettings, LoggerSettings
from anvilrl.updaters.evolution import NoisyGradientAscent


class MockDeepAgent(BaseDeepAgent):
    def _fit():
        pass


class MockEvolutionAgent(BaseEvolutionAgent):
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
individual = Individual(space=env.action_space)

deep_agent = MockDeepAgent(
    env=env,
    model=model,
    buffer_class=ReplayBuffer,
    explorer_settings=ExplorerSettings(start_steps=0),
    logger_settings=LoggerSettings(tensorboard_log_path="runs/tests"),
)
vec_deep_agent = MockDeepAgent(
    env=envs,
    model=model,
    buffer_class=ReplayBuffer,
    explorer_settings=ExplorerSettings(start_steps=0),
    logger_settings=LoggerSettings(tensorboard_log_path="runs/tests"),
)
evolution_agent = MockEvolutionAgent(
    env=envs,
    model=individual,
    updater_class=NoisyGradientAscent,
    buffer_class=ReplayBuffer,
)

shutil.rmtree("runs/tests")


def test_step_env():
    env.seed(0)
    observation = env.reset()
    action = model(observation).detach().numpy()
    expected_next_obs, _, _, _ = env.step(action)
    actual_next_obs = deep_agent.step_env(observation)

    assert isinstance(actual_next_obs, np.ndarray)

    observation = envs.reset()
    action = model(observation).detach().numpy()
    expected_next_obs, _, _, _ = env.step(action)
    actual_next_obs = vec_deep_agent.step_env(observation)

    assert isinstance(actual_next_obs, np.ndarray)
