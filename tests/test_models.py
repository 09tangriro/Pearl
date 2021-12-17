import copy

import gym
import numpy as np
import pytest
import torch as T

from anvilrl.models import (
    Actor,
    ActorCritic,
    ActorCriticWithTargets,
    Critic,
    EpsilonGreedyActor,
    TwinActorCritic,
)
from anvilrl.models.actor_critics import DeepIndividual, Individual
from anvilrl.models.encoders import (
    DictEncoder,
    FlattenEncoder,
    IdentityEncoder,
    MLPEncoder,
)
from anvilrl.models.heads import (
    CategoricalHead,
    ContinuousQHead,
    DeterministicHead,
    DiagGaussianHead,
    DiscreteQHead,
    ValueHead,
)
from anvilrl.models.torsos import MLP
from anvilrl.models.utils import trainable_parameters


def test_mlp():
    input = T.Tensor([1, 1])
    model = MLP([2, 1])
    output = model(input)
    assert output.shape == (1,)


@pytest.mark.parametrize(
    "encoder_class", [IdentityEncoder, FlattenEncoder, MLPEncoder, DictEncoder]
)
def test_encoder(encoder_class):
    input = T.Tensor([[2, 2], [1, 1]])
    if encoder_class == DictEncoder:
        input = {"observation": T.Tensor([[2, 2], [1, 1]])}
    if encoder_class == MLPEncoder:
        encoder = encoder_class(input_size=2, output_size=2)
    elif encoder_class == DictEncoder:
        encoder = encoder_class(labels=["observation"])
    else:
        encoder = encoder_class()
    output = encoder(input)
    if isinstance(encoder, (IdentityEncoder)):
        assert T.equal(input, output)
    elif encoder_class == DictEncoder:
        assert T.equal(input["observation"], output)
    elif isinstance(encoder, FlattenEncoder):
        assert len(output.shape) == 2


@pytest.mark.parametrize("head_class", [ValueHead, ContinuousQHead, DiscreteQHead])
@pytest.mark.parametrize("input_shape", [5, (5,)])
def test_critic_head(head_class, input_shape):
    input = T.Tensor([1, 1, 1, 1, 1])
    if head_class == DiscreteQHead:
        head = head_class(input_shape=input_shape, output_shape=(2,))
    else:
        head = head_class(input_shape=input_shape)

    output = head(input)

    if head_class == DiscreteQHead:
        assert output.shape == (2,)
    else:
        assert output.shape == (1,)


@pytest.mark.parametrize(
    "head_class", [DeterministicHead, CategoricalHead, DiagGaussianHead]
)
@pytest.mark.parametrize("input_shape", [5, (5,)])
def test_actor_head(head_class, input_shape):
    input = T.Tensor([1, 1, 1, 1, 1])
    if head_class == DeterministicHead:
        head = head_class(input_shape, action_shape=2)
    else:
        head = head_class(input_shape=input_shape, action_size=2)

    output = head(input)

    if head_class == CategoricalHead:
        assert output.shape == T.Size([])
    else:
        assert output.shape == (2,)


def test_critic():
    input = T.Tensor([1, 1, 1, 1, 1])
    encoder = IdentityEncoder()
    torso = MLP([5, 5])
    head = ValueHead(input_shape=5)

    critic = Critic(encoder, torso, head)

    output = critic(input)
    assert output.shape == (1,)


def test_actor():
    input = T.Tensor([1, 1, 1, 1, 1])
    encoder = IdentityEncoder()
    torso = MLP([5, 5])
    head = DeterministicHead(input_shape=5, action_shape=1)

    actor = Actor(encoder, torso, head)

    output = actor(input)
    assert output.shape == (1,)


@pytest.mark.parametrize(
    "actor_critic_class", [ActorCritic, ActorCriticWithTargets, TwinActorCritic]
)
def test_actor_critic_shared_arch(actor_critic_class):
    input = T.tensor([1, 1], dtype=T.float32)
    encoder = IdentityEncoder()
    actor = Actor(
        encoder=encoder,
        torso=MLP([2, 3, 2]),
        head=DeterministicHead(input_shape=2, action_shape=1, activation_fn=None),
    )

    critic = Critic(
        encoder=encoder,
        torso=MLP([2, 2]),
        head=ValueHead(input_shape=2, activation_fn=None),
    )

    actor_critic = actor_critic_class(actor=actor, critic=critic)

    assert actor_critic.actor and actor_critic.critic

    if actor_critic_class == TwinActorCritic:
        assert actor_critic.forward_critic(input) == (
            actor_critic.target_critic(input),
            actor_critic.target_critic2(input),
        )
        assert actor_critic(input) == actor_critic.forward_target_actor(input)

    if actor_critic_class == ActorCriticWithTargets:
        assert actor_critic.forward_critic(input) == actor_critic.forward_target_critic(
            input
        )
        assert actor_critic(input) == actor_critic.forward_target_actor(input)


def test_epsilon_greedy_actor():
    action_size = 2
    observation_size = 2

    encoder = IdentityEncoder()
    torso = MLP(layer_sizes=[observation_size, 16, 16], activation_fn=T.nn.ReLU)
    head = DiscreteQHead(input_shape=16, output_shape=action_size)

    actor = EpsilonGreedyActor(
        critic_encoder=encoder, critic_torso=torso, critic_head=head
    )

    input = T.tensor([[1, 1], [2, 2]], dtype=T.float32)

    random_output = actor(input)

    actor = EpsilonGreedyActor(
        critic_encoder=encoder, critic_torso=torso, critic_head=head, start_epsilon=0
    )

    max_q_output = actor(input)

    assert random_output.shape == max_q_output.shape


def test_trainable_parameters():
    mlp = MLP([2, 3])

    weights, biases = trainable_parameters(mlp)
    assert weights.shape == T.Size([3, 2])
    assert biases.shape == T.Size([3])

    all_params = T.cat((T.flatten(weights), biases))

    assert all_params.shape == T.Size([9])


def test_deep_individual():
    T.manual_seed(0)
    model = DeepIndividual(
        encoder=IdentityEncoder(),
        torso=MLP([2]),
        head=DeterministicHead(input_shape=2, action_shape=1),
    )

    actual_state = model.numpy()
    expected_state = np.array([-0.00529397, 0.37932295, -0.58198076])

    np.testing.assert_array_almost_equal(actual_state, expected_state)

    expected_state = np.array([0.1, 0.2, 0.3])
    expected_model = copy.deepcopy(model.model)
    model.set_state(expected_state)
    actual_state = model.numpy()
    assert model.model != expected_model
    np.testing.assert_array_almost_equal(actual_state, expected_state)


def test_individual():
    env = gym.make("CartPole-v0")
    T.manual_seed(0)
    expected_state = env.action_space.sample()
    model = Individual(space=env.action_space, state=expected_state)

    actual_state = model.numpy()
    np.testing.assert_array_almost_equal(actual_state, expected_state)

    expected_state = env.action_space.sample()
    model.set_state(expected_state)

    np.testing.assert_array_almost_equal(model.numpy(), expected_state)

    actual_state = model(env.observation_space.sample())
    np.testing.assert_array_almost_equal(actual_state, expected_state)
