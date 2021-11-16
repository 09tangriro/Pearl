import gym
import numpy as np
import pytest
import torch as T

from anvilrl.models import Actor, ActorCritic, Critic
from anvilrl.models.encoders import IdentityEncoder, MLPEncoder
from anvilrl.models.heads import DiagGaussianPolicyHead, ValueHead
from anvilrl.models.torsos import MLP
from anvilrl.updaters.actors import (
    DeterministicPolicyGradient,
    PolicyGradient,
    ProximalPolicyClip,
    SoftPolicyGradient,
)
from anvilrl.updaters.critics import QRegression, ValueRegression
from anvilrl.updaters.random_search import EvolutionaryUpdater

############################### SET UP MODELS ###############################

encoder_critic = IdentityEncoder()
encoder_critic_continuous = MLPEncoder(input_size=3, output_size=2)
encoder_actor = IdentityEncoder()
torso_critic = MLP(layer_sizes=[2, 2])
torso_actor = MLP(layer_sizes=[2, 2])
head_actor = DiagGaussianPolicyHead(input_shape=2, action_size=1)
head_critic = ValueHead(input_shape=2, activation_fn=None)

actor = Actor(encoder=encoder_actor, torso=torso_actor, head=head_actor)
critic = Critic(encoder=encoder_critic, torso=torso_critic, head=head_critic)
continuous_critic = Critic(
    encoder=encoder_critic_continuous, torso=torso_critic, head=head_critic
)
continuous_critic_shared = Critic(
    encoder=encoder_critic_continuous, torso=torso_actor, head=head_critic
)
critic_shared_encoder = Critic(
    encoder=encoder_actor, torso=torso_critic, head=head_critic
)
critic_shared = Critic(encoder=encoder_actor, torso=torso_actor, head=head_critic)

actor_critic = ActorCritic(actor=actor, critic=critic)
actor_critic_shared_encoder = ActorCritic(actor=actor, critic=critic_shared_encoder)
actor_critic_shared = ActorCritic(actor=actor, critic=critic_shared)
continuous_actor_critic = ActorCritic(actor=actor, critic=continuous_critic)
continuous_actor_critic_shared = ActorCritic(
    actor=actor, critic=continuous_critic_shared
)


def assert_same_distribution(
    dist1: T.distributions.Distribution, dist2: T.distributions.Distribution
) -> bool:
    if T.equal(dist1.loc, dist2.loc) and T.equal(dist1.scale, dist2.scale):
        return True
    else:
        return False


############################### TEST ACTOR UPDATERS ###############################


@pytest.mark.parametrize(
    "model", [actor, actor_critic, actor_critic_shared_encoder, actor_critic_shared]
)
def test_policy_gradient(model):
    observation = T.rand(2)
    out_before = model.get_action_distribution(observation)
    if model != actor:
        with T.no_grad():
            critic_before = model.forward_critic(observation)

    updater = PolicyGradient(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
        actions=T.rand(1),
        advantages=T.rand(1),
    )

    out_after = model.get_action_distribution(observation)
    if model != actor:
        with T.no_grad():
            critic_after = model.forward_critic(observation)

    assert not assert_same_distribution(out_after, out_before)
    if model == actor_critic or model == actor_critic_shared_encoder:
        assert critic_after == critic_before
    if model == actor_critic_shared:
        assert critic_after != critic_before


@pytest.mark.parametrize(
    "model", [actor, actor_critic, actor_critic_shared_encoder, actor_critic_shared]
)
def test_proximal_policy_clip(model):
    observation = T.rand(2)
    out_before = model.get_action_distribution(observation)
    if model != actor:
        with T.no_grad():
            critic_before = model.forward_critic(observation)

    updater = ProximalPolicyClip(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
        actions=T.rand(1),
        advantages=T.rand(1),
        old_log_probs=T.rand(1),
    )

    out_after = model.get_action_distribution(observation)
    if model != actor:
        with T.no_grad():
            critic_after = model.forward_critic(observation)

    assert not assert_same_distribution(out_after, out_before)
    if model == actor_critic or model == actor_critic_shared_encoder:
        assert critic_after == critic_before
    if model == actor_critic_shared:
        assert critic_after != critic_before


@pytest.mark.parametrize(
    "model", [continuous_actor_critic, continuous_actor_critic_shared]
)
def test_deterministic_policy_gradient(model):
    observation = T.rand(2)
    action = T.rand(1)
    out_before = model(observation)
    with T.no_grad():
        critic_before = model.forward_critic(observation, action)

    updater = DeterministicPolicyGradient(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
    )

    out_after = model(observation)
    with T.no_grad():
        critic_after = model.forward_critic(observation, action)

    assert out_after != out_before
    if model == continuous_actor_critic:
        assert critic_after == critic_before
    if model == continuous_actor_critic_shared:
        assert critic_after != critic_before


@pytest.mark.parametrize(
    "model", [continuous_actor_critic, continuous_actor_critic_shared]
)
def test_soft_policy_gradient(model):
    observation = T.rand(2)
    action = T.rand(1)
    out_before = model.get_action_distribution(observation)
    with T.no_grad():
        critic_before = model.forward_critic(observation, action)

    updater = SoftPolicyGradient(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
    )

    out_after = model.get_action_distribution(observation)
    with T.no_grad():
        critic_after = model.forward_critic(observation, action)

    assert not assert_same_distribution(out_after, out_before)
    if model == continuous_actor_critic:
        assert critic_after == critic_before
    if model == continuous_actor_critic_shared:
        assert critic_after != critic_before


############################### TEST CRITIC UPDATERS ###############################


@pytest.mark.parametrize(
    "model", [actor_critic, actor_critic_shared_encoder, actor_critic_shared]
)
def test_value_regression(model):
    observation = T.rand(2)
    returns = T.rand(1)
    out_before = model.forward_critic(observation)
    with T.no_grad():
        actor_before = model.get_action_distribution(observation)

    updater = ValueRegression(max_grad=0.5)

    updater(model, observation, returns)

    out_after = model.forward_critic(observation)
    with T.no_grad():
        actor_after = model.get_action_distribution(observation)

    assert out_after != out_before
    if model == actor_critic_shared:
        assert not assert_same_distribution(actor_before, actor_after)
    else:
        assert assert_same_distribution(actor_before, actor_after)


@pytest.mark.parametrize(
    "model", [actor_critic, actor_critic_shared_encoder, actor_critic_shared]
)
def test_q_regression(model):
    observation = T.rand(2)
    returns = T.rand(1)
    out_before = model.forward_critic(observation)
    with T.no_grad():
        actor_before = model.get_action_distribution(observation)

    updater = QRegression(max_grad=0.5)

    updater(model, observation, returns)

    out_after = model.forward_critic(observation)
    with T.no_grad():
        actor_after = model.get_action_distribution(observation)

    assert out_after != out_before
    if model == actor_critic_shared:
        assert not assert_same_distribution(actor_before, actor_after)
    else:
        assert assert_same_distribution(actor_before, actor_after)


############################### TEST RANDOM SEARCH UPDATERS ###############################


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


POPULATION_SIZE = 10000
env = gym.vector.SyncVectorEnv([lambda: Sphere() for _ in range(POPULATION_SIZE)])


def test_evolutionary_updater():
    np.random.seed(0)

    # Assert population stats
    updater = EvolutionaryUpdater(env)
    population = updater.initialize_population(starting_point=np.array([10, 10]))
    np.testing.assert_allclose(np.std(population, axis=0), np.ones(2), rtol=0.1)
    np.testing.assert_allclose(
        np.mean(population, axis=0), np.array([10, 10]), rtol=0.1
    )

    # Test call
    _, rewards, _, _ = env.step(population)
    updater(rewards=rewards, lr=1)
    new_population = updater.population
    assert new_population.shape == (POPULATION_SIZE, 2)
    np.testing.assert_allclose(np.std(new_population, axis=0), np.ones(2), rtol=0.1)
    np.testing.assert_array_less(updater.mean, np.array([10, 10]))
