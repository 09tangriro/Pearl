import copy
from typing import Union

import gym
import numpy as np
import pytest
import torch as T

from pearll.models import Actor, ActorCritic, Critic, Dummy
from pearll.models.actor_critics import Model
from pearll.models.encoders import IdentityEncoder, MLPEncoder
from pearll.models.heads import BoxHead, DiagGaussianHead, ValueHead
from pearll.models.torsos import MLP
from pearll.settings import PopulationSettings
from pearll.signal_processing import (
    crossover_operators,
    mutation_operators,
    selection_operators,
)
from pearll.updaters.actors import (
    DeterministicPolicyGradient,
    PolicyGradient,
    ProximalPolicyClip,
    SoftPolicyGradient,
)
from pearll.updaters.critics import (
    ContinuousQRegression,
    DiscreteQRegression,
    ValueRegression,
)
from pearll.updaters.environment import DeepRegression
from pearll.updaters.evolution import GeneticUpdater, NoisyGradientAscent

############################### SET UP MODELS ###############################

encoder_critic = IdentityEncoder()
encoder_critic_continuous = MLPEncoder(input_size=3, output_size=2)
encoder_actor = IdentityEncoder()
torso_critic = MLP(layer_sizes=[2, 2])
torso_actor = MLP(layer_sizes=[2, 2])
head_actor = DiagGaussianHead(input_shape=2, action_size=1)
head_critic = ValueHead(input_shape=2, activation_fn=None)

actor = Actor(encoder=encoder_actor, torso=torso_actor, head=head_actor)
critic = Critic(encoder=encoder_critic, torso=torso_critic, head=head_critic)
continuous_critic = Critic(
    encoder=encoder_critic_continuous, torso=torso_critic, head=head_critic
)
continuous_critic_shared = Critic(
    encoder=encoder_critic_continuous, torso=torso_actor, head=head_critic
)
critic_shared = Critic(encoder=encoder_actor, torso=torso_actor, head=head_critic)

actor_critic = ActorCritic(actor=actor, critic=critic)
actor_critic_shared = ActorCritic(actor=actor, critic=critic_shared)
continuous_actor_critic = ActorCritic(actor=actor, critic=continuous_critic)
continuous_actor_critic_shared = ActorCritic(
    actor=actor, critic=continuous_critic_shared
)
marl = ActorCritic(
    actor=actor,
    critic=critic,
    population_settings=PopulationSettings(
        actor_population_size=2, critic_population_size=2
    ),
)
marl_shared = ActorCritic(
    actor=actor,
    critic=critic_shared,
    population_settings=PopulationSettings(
        actor_population_size=2, critic_population_size=2
    ),
)
marl_continuous = ActorCritic(
    actor=actor,
    critic=continuous_critic,
    population_settings=PopulationSettings(
        actor_population_size=2, critic_population_size=2
    ),
)
marl_shared_continuous = ActorCritic(
    actor=actor,
    critic=continuous_critic_shared,
    population_settings=PopulationSettings(
        actor_population_size=2, critic_population_size=2
    ),
)

T.manual_seed(0)
np.random.seed(0)


def same_distribution(
    dist1: T.distributions.Distribution, dist2: T.distributions.Distribution
) -> bool:
    return T.equal(dist1.loc, dist2.loc) and T.equal(dist1.scale, dist2.scale)


############################### TEST ACTOR UPDATERS ###############################


@pytest.mark.parametrize(
    "model", [actor, actor_critic, actor_critic_shared, marl, marl_shared]
)
def test_policy_gradient(model: Union[Actor, ActorCritic]):
    observation = T.rand(2)
    if model != actor:
        with T.no_grad():
            observation = observation.repeat(model.num_actors, 1)
            critic_before = model.forward_critics(observation)
    out_before = model.action_distribution(observation)

    updater = PolicyGradient(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
        actions=T.rand(1),
        advantages=T.rand(1),
    )

    out_after = model.action_distribution(observation)
    if model != actor:
        with T.no_grad():
            critic_after = model.forward_critics(observation)

    assert not same_distribution(out_after, out_before)
    if model == actor_critic or model == marl:
        assert T.equal(critic_before, critic_after)
    if model == actor_critic_shared or model == marl_shared:
        assert not T.equal(critic_before, critic_after)


@pytest.mark.parametrize(
    "model", [actor, actor_critic, actor_critic_shared, marl, marl_shared]
)
def test_proximal_policy_clip(model: Union[Actor, ActorCritic]):
    observation = T.rand(2)
    if model != actor:
        with T.no_grad():
            observation = observation.repeat(model.num_actors, 1)
            critic_before = model.forward_critics(observation)
    out_before = model.action_distribution(observation)

    updater = ProximalPolicyClip(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
        actions=T.rand(1),
        advantages=T.rand(1),
        old_log_probs=T.rand(1),
    )

    out_after = model.action_distribution(observation)
    if model != actor:
        with T.no_grad():
            critic_after = model.forward_critics(observation)

    assert not same_distribution(out_after, out_before)
    if model == actor_critic or model == marl:
        assert T.equal(critic_before, critic_after)
    if model == actor_critic_shared or model == marl_shared:
        assert not T.equal(critic_before, critic_after)


@pytest.mark.parametrize(
    "model",
    [
        continuous_actor_critic,
        continuous_actor_critic_shared,
        marl_continuous,
        marl_shared_continuous,
    ],
)
def test_deterministic_policy_gradient(model: ActorCritic):
    observation = T.rand(2).repeat(model.num_actors, 1)
    action = model(observation)
    with T.no_grad():
        critic_before = model.forward_critics(observation, action)

    updater = DeterministicPolicyGradient(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
    )

    out_after = model(observation)
    with T.no_grad():
        critic_after = model.forward_critics(observation, action)

    assert not T.equal(action, out_after)
    if model == continuous_actor_critic or model == marl_continuous:
        assert T.equal(critic_before, critic_after)
    if model == continuous_actor_critic_shared or model == marl_shared_continuous:
        assert not T.equal(critic_before, critic_after)


@pytest.mark.parametrize(
    "model",
    [
        continuous_actor_critic,
        continuous_actor_critic_shared,
        marl_continuous,
        marl_shared_continuous,
    ],
)
def test_soft_policy_gradient(model: ActorCritic):
    observation = T.rand(2).repeat(model.num_actors, 1)
    action = model(observation)
    out_before = model.action_distribution(observation)
    with T.no_grad():
        critic_before = model.forward_critics(observation, action)

    updater = SoftPolicyGradient(max_grad=0.5)

    updater(
        model=model,
        observations=observation,
    )

    out_after = model.action_distribution(observation)
    with T.no_grad():
        critic_after = model.forward_critics(observation, action)

    assert not same_distribution(out_after, out_before)
    if model == continuous_actor_critic or model == marl_continuous:
        assert T.equal(critic_before, critic_after)
    if model == continuous_actor_critic_shared or model == marl_shared_continuous:
        assert not T.equal(critic_before, critic_after)


############################### TEST CRITIC UPDATERS ###############################


@pytest.mark.parametrize(
    "model", [critic, actor_critic, actor_critic_shared, marl, marl_shared]
)
def test_value_regression(model: Union[Critic, ActorCritic]):
    observation = T.rand(2)
    returns = T.rand(1)
    if model != critic:
        observation = observation.repeat(model.num_critics, 1)
        with T.no_grad():
            actor_before = model.action_distribution(observation)
        out_before = model.forward_critics(observation)
    else:
        out_before = model(observation)

    updater = ValueRegression(max_grad=0.5)

    updater(model, observation, returns)

    if model != critic:
        out_after = model.forward_critics(observation)
        with T.no_grad():
            actor_after = model.action_distribution(observation)
    else:
        out_after = model(observation)

    assert not T.equal(out_after, out_before)
    if model == actor_critic_shared or model == marl_shared:
        assert not same_distribution(actor_before, actor_after)
    elif model == actor_critic or model == marl:
        assert same_distribution(actor_before, actor_after)


@pytest.mark.parametrize("model", [critic, actor_critic, actor_critic_shared])
def test_discrete_q_regression(model: Union[Critic, ActorCritic]):
    observation = T.rand(1, 2)
    actions = T.randint(0, 1, (1, 1))
    returns = T.rand(1)
    if model == critic:
        out_before = model(observation)
    else:
        out_before = model.forward_critics(observation)
        with T.no_grad():
            actor_before = model.action_distribution(observation)

    updater = DiscreteQRegression(max_grad=0.5)

    updater(model, observation, returns, actions)

    if model == critic:
        out_after = model(observation)
    else:
        out_after = model.forward_critics(observation)
        with T.no_grad():
            actor_after = model.action_distribution(observation)

    assert not T.equal(out_before, out_after)
    if model == actor_critic_shared:
        assert not same_distribution(actor_before, actor_after)
    elif model == actor_critic:
        assert same_distribution(actor_before, actor_after)


@pytest.mark.parametrize(
    "model",
    [continuous_actor_critic, continuous_actor_critic_shared, continuous_critic],
)
def test_continuous_q_regression(model: Union[Critic, ActorCritic]):
    observation = T.rand(1, 2)
    actions = T.rand(1, 1)
    returns = T.rand(1)
    if model == continuous_critic:
        out_before = model(observation, actions)
    else:
        out_before = model.forward_critics(observation, actions)
        with T.no_grad():
            actor_before = model.action_distribution(observation)

    updater = ContinuousQRegression(max_grad=0.5)

    updater(model, observation, actions, returns)

    if model == continuous_critic:
        out_after = model(observation, actions)
    else:
        out_after = model.forward_critics(observation, actions)
        with T.no_grad():
            actor_after = model.action_distribution(observation)

    assert out_after != out_before
    if model == continuous_actor_critic_shared:
        assert not same_distribution(actor_before, actor_after)
    elif model == continuous_actor_critic:
        assert same_distribution(actor_before, actor_after)


############################### TEST EVOLUTION UPDATERS ###############################


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


class DiscreteSphere(gym.Env):
    """
    Discrete Sphere(1) function for testing ES agent.
    """

    def __init__(self):
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, action):
        return 0, -(action ** 2), False, {}

    def reset(self):
        return 0


POPULATION_SIZE = 100
env_continuous = gym.vector.SyncVectorEnv(
    [lambda: Sphere() for _ in range(POPULATION_SIZE)]
)
env_discrete = gym.vector.SyncVectorEnv(
    [lambda: DiscreteSphere() for _ in range(POPULATION_SIZE)]
)


def test_evolutionary_updater_continuous():
    actor_continuous = Dummy(
        space=env_continuous.single_action_space, state=np.array([10, 10])
    )
    critic = Dummy(space=env_continuous.single_action_space)
    model_continuous = ActorCritic(
        actor=actor_continuous,
        critic=critic,
        population_settings=PopulationSettings(
            actor_population_size=POPULATION_SIZE, actor_distribution="normal"
        ),
    )

    # ASSERT POPULATION STATS
    updater = NoisyGradientAscent(model_continuous)
    # make sure starting population mean is correct
    np.testing.assert_allclose(
        np.mean(model_continuous.numpy_actors(), axis=0), np.array([10, 10]), rtol=0.1
    )

    # TEST CALL
    old_model = copy.deepcopy(model_continuous)
    old_population = model_continuous.numpy_actors()
    action = model_continuous(np.zeros(POPULATION_SIZE))
    _, rewards, _, _ = env_continuous.step(action)
    scaled_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    optimization_direction = np.dot(updater.normal_dist.T, scaled_rewards)
    log = updater(learning_rate=0.01, optimization_direction=optimization_direction)
    new_population = model_continuous.numpy_actors()
    assert log.divergence > 0
    # make sure the nerual network has been updated by the updater
    assert model_continuous != old_model
    assert np.not_equal(old_population, new_population).any()
    # make sure the network mean has been updated by the updater
    np.testing.assert_array_equal(model_continuous.mean_actor, updater.mean)
    # make sure the new popualtion has the correct std
    np.testing.assert_allclose(
        np.std(model_continuous.numpy_actors(), axis=0), np.ones(2), rtol=0.1
    )
    # make sure the update direction is correct
    np.testing.assert_array_less(
        np.mean(model_continuous.numpy_actors(), axis=0), np.array([10, 10])
    )


def test_evolutionary_updater_discrete():
    actor_discrete = Dummy(space=env_discrete.single_action_space, state=np.array([5]))
    critic = Dummy(space=env_discrete.single_action_space)
    model_discrete = ActorCritic(
        actor=actor_discrete,
        critic=critic,
        population_settings=PopulationSettings(
            actor_population_size=POPULATION_SIZE, actor_distribution="normal"
        ),
    )

    # ASSERT POPULATION STATS
    updater = NoisyGradientAscent(model_discrete)
    # make sure the population has been discretized
    assert np.issubdtype(model_discrete.numpy_actors().dtype, np.integer)
    # make sure starting population mean is correct
    np.testing.assert_allclose(
        np.mean(model_discrete.numpy_actors(), axis=0), np.array([5]), rtol=0.1
    )

    # Test call
    old_model = copy.deepcopy(model_discrete)
    old_population = model_discrete.numpy_actors()
    action = model_discrete(np.zeros(POPULATION_SIZE))
    _, rewards, _, _ = env_discrete.step(action)
    scaled_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    optimization_direction = np.dot(updater.normal_dist.T, scaled_rewards)
    log = updater(learning_rate=1e-5, optimization_direction=optimization_direction)
    new_population = model_discrete.numpy_actors()
    assert log.divergence > 0
    # make sure the nerual network has been updated by the updater
    assert model_discrete != old_model
    assert np.not_equal(old_population, new_population).any()
    # make sure the population has been discretized
    assert np.issubdtype(new_population.dtype, np.integer)
    # make sure the network mean has been updated by the updater
    np.testing.assert_array_equal(model_discrete.mean_actor, updater.mean)
    # make sure the new popualtion has the correct std
    np.testing.assert_allclose(np.std(new_population, axis=0), np.ones(1), rtol=0.1)
    # make sure the update direction is correct
    np.testing.assert_array_less(model_discrete.mean_actor, np.array([5]))


def test_genetic_updater_continuous():
    actor_continuous = Dummy(
        space=env_continuous.single_action_space, state=np.array([10, 10])
    )
    critic = Dummy(space=env_continuous.single_action_space)
    model_continuous = ActorCritic(
        actor=actor_continuous,
        critic=critic,
        population_settings=PopulationSettings(
            actor_population_size=POPULATION_SIZE, actor_distribution="normal"
        ),
    )

    # Assert population stats
    updater = GeneticUpdater(model_continuous)
    np.testing.assert_allclose(
        np.mean(model_continuous.numpy_actors(), axis=0), np.array([10, 10]), rtol=0.1
    )

    # Test call
    old_model = copy.deepcopy(model_continuous)
    old_population = model_continuous.numpy_actors()
    action = model_continuous(np.zeros(POPULATION_SIZE))
    _, rewards, _, _ = env_continuous.step(action)
    log = updater(
        rewards=rewards,
        selection_operator=selection_operators.roulette_selection,
        crossover_operator=crossover_operators.one_point_crossover,
        mutation_operator=mutation_operators.uniform_mutation,
    )
    new_population = model_continuous.numpy_actors()
    assert log.divergence > 0
    assert model_continuous != old_model
    assert np.not_equal(old_population, new_population).any()
    np.testing.assert_array_less(np.min(new_population, axis=0), np.array([10, 10]))


def test_genetic_updater_discrete():
    actor_discrete = Dummy(space=env_discrete.single_action_space, state=np.array([5]))
    critic = Dummy(space=env_discrete.single_action_space)
    model_discrete = ActorCritic(
        actor=actor_discrete,
        critic=critic,
        population_settings=PopulationSettings(
            actor_population_size=POPULATION_SIZE, actor_distribution="uniform"
        ),
    )

    # Assert population stats
    updater = GeneticUpdater(model_discrete)
    old_model = copy.deepcopy(model_discrete)
    old_population = model_discrete.numpy_actors()
    assert np.issubdtype(old_population.dtype, np.integer)
    np.testing.assert_allclose(np.mean(old_population, axis=0), np.array([5]), rtol=0.2)

    # Test call
    action = model_discrete(np.zeros(POPULATION_SIZE))
    _, rewards, _, _ = env_discrete.step(action)
    log = updater(
        rewards=rewards,
        selection_operator=selection_operators.roulette_selection,
        crossover_operator=crossover_operators.one_point_crossover,
        mutation_operator=mutation_operators.uniform_mutation,
    )
    new_population = model_discrete.numpy_actors()

    assert np.issubdtype(new_population.dtype, np.integer)
    assert log.divergence > 0
    assert model_discrete != old_model
    assert np.not_equal(old_population, new_population).any()
    np.testing.assert_array_less(np.min(new_population, axis=0), np.array([5]))


############################### TEST ENVIRONMENT UPDATERS ###############################


def test_deep_env_updater():
    T.manual_seed(0)
    np.random.seed(0)
    encoder = IdentityEncoder()
    torso = MLP(layer_sizes=[2, 1, 1])
    head = BoxHead(input_shape=1, space_shape=2)
    deep_model = Model(encoder=encoder, torso=torso, head=head)
    updater = DeepRegression()

    observations = T.Tensor([[1], [3]])
    actions = T.Tensor([[1], [2]])
    targets = T.Tensor([[1], [2]])
    log = updater(deep_model, observations, actions, targets)
    assert log.loss == 1.8426234722137451
