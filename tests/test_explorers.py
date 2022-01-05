import gym
import pytest

from anvilrl.explorers import BaseExplorer, GaussianExplorer
from anvilrl.models.actor_critics import Actor, ActorCritic, Critic
from anvilrl.models.encoders import IdentityEncoder
from anvilrl.models.heads import DeterministicHead, ValueHead
from anvilrl.models.torsos import MLP
from anvilrl.models.utils import get_mlp_size
from anvilrl.settings import PopulationSettings


@pytest.mark.parametrize("explorer_class", [BaseExplorer, GaussianExplorer])
def test_explorer(explorer_class):
    ###############################
    # Single agent explorer tests
    ###############################
    env = gym.make("MountainCarContinuous-v0")
    observation = env.reset()
    observation_size = get_mlp_size(env.observation_space.shape)
    encoder = IdentityEncoder()
    torso = MLP([observation_size, 10, 10])
    actor_head = DeterministicHead(
        input_shape=10, action_shape=env.action_space.shape, activation_fn=None
    )
    critic_head = ValueHead(input_shape=10, activation_fn=None)
    actor = Actor(encoder=encoder, torso=torso, head=actor_head)
    critic = Critic(encoder=encoder, torso=torso, head=critic_head)
    model = ActorCritic(actor=actor, critic=critic)
    explorer = explorer_class(action_space=env.action_space)

    # uniform exploration
    action = explorer(model=actor, observation=observation, step=0)
    assert abs(action[0]) <= 1
    action = explorer(model=model, observation=observation, step=0)
    assert abs(action[0]) <= 1

    # model exploration
    action = explorer(model=actor, observation=observation, step=50e3)
    assert abs(action[0]) <= 1
    action = explorer(model=model, observation=observation, step=50e3)
    assert abs(action[0]) <= 1

    ###############################
    # Multi agent explorer tests
    ###############################
    env = gym.vector.make("MountainCarContinuous-v0", num_envs=2)
    observation = env.reset()
    model = ActorCritic(
        actor=actor,
        critic=critic,
        population_settings=PopulationSettings(
            actor_population_size=2, critic_population_size=2
        ),
    )
    explorer = explorer_class(action_space=env.action_space)

    # uniform exploration
    actions = explorer(model=model, observation=observation, step=0)
    for a in actions:
        assert abs(a[0]) <= 1

    # model exploration
    actions = explorer(model=model, observation=observation, step=50e3)
    assert actions.shape == (2,)
