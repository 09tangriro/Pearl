import gym
import pytest

from anvil_rl.explorers import BaseExplorer, GaussianExplorer
from anvil_rl.models.actor_critics import Actor
from anvil_rl.models.encoders import IdentityEncoder
from anvil_rl.models.heads import DeterministicPolicyHead
from anvil_rl.models.torsos import MLP
from anvil_rl.models.utils import get_mlp_size


@pytest.mark.parametrize("explorer_class", [BaseExplorer, GaussianExplorer])
def test_base_explorer(explorer_class):
    num_steps = 1000
    env = gym.make("MountainCarContinuous-v0")
    observation = env.reset()
    observation_size = get_mlp_size(env.observation_space.shape)
    encoder = IdentityEncoder()
    torso = MLP([observation_size, 10, 10])
    head = DeterministicPolicyHead(
        input_shape=10, action_shape=env.action_space.shape, activation_fn=None
    )
    actor = Actor(encoder=encoder, torso=torso, head=head)
    explorer = explorer_class(action_space=env.action_space)

    # uniform exploration
    for _ in range(num_steps):
        action = explorer(actor=actor, observation=observation, step=0)
        assert abs(action[0]) <= 1

    action = explorer(actor=actor, observation=observation, step=50e3)
    assert abs(action[0]) <= 1
