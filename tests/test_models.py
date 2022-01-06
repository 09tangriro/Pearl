import gym
import numpy as np
import pytest
import torch as T

from anvilrl.models import (
    Actor,
    ActorCritic,
    Critic,
    DummyActor,
    DummyCritic,
    EpsilonGreedyActor,
)
from anvilrl.models.encoders import (
    CNNEncoder,
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
from anvilrl.settings import PopulationSettings

T.manual_seed(0)
np.random.seed(0)


def test_mlp():
    input = T.Tensor([1, 1])
    model = MLP([2, 1])
    output = model(input)
    assert output.shape == (1,)


@pytest.mark.parametrize("encoder_class", [IdentityEncoder, FlattenEncoder, MLPEncoder])
def test_encoder(encoder_class):
    input = T.Tensor([[2, 2], [1, 1]])
    if encoder_class == MLPEncoder:
        encoder = encoder_class(input_size=2, output_size=2)
    elif encoder_class == DictEncoder:
        encoder = encoder_class(labels=["observation"])
    else:
        encoder = encoder_class()
    output = encoder(input)
    if isinstance(encoder, (IdentityEncoder)):
        assert T.equal(input, output)
    else:
        assert len(output.shape) == 2


def test_dict_encoder():
    input = {
        "observation": T.Tensor([[2, 2], [1, 1]]),
        "action": T.Tensor([[1, 1], [2, 2]]),
    }
    encoder = DictEncoder(labels=["observation", "action"])
    actual_output = encoder(input)
    expected_output = T.cat([input["observation"], input["action"]], dim=1)
    assert T.equal(expected_output, actual_output)


def test_cnn_encoder():
    space = gym.spaces.Box(low=0, high=255, shape=(1, 64, 64))
    encoder = CNNEncoder(observation_space=space)

    input = T.normal(0, 1, (1, 1, 64, 64))
    output = encoder(input)

    assert output.shape == (1, 512)


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

    critic = Critic(encoder, torso, head, create_target=True)

    online_output = critic(input)
    target_output = critic.forward_target(input)
    assert T.equal(online_output, target_output)

    state_shape = critic.numpy().shape
    new_state = np.random.rand(*state_shape)
    critic.set_state(new_state)
    new_output = critic(input)
    assert not T.equal(online_output, new_output)
    np.testing.assert_array_equal(critic.numpy(), new_state)

    online_output = critic(input)
    new_target_output = critic.forward_target(input)
    assert not T.equal(online_output, new_target_output)
    assert T.equal(target_output, new_target_output)

    critic.update_targets()
    update_target_output = critic.forward_target(input)
    assert not T.equal(new_target_output, update_target_output)
    assert not T.equal(online_output, update_target_output)

    critic.assign_targets()
    target_output = critic.forward_target(input)

    assert T.equal(online_output, target_output)


def test_actor():
    input = T.Tensor([1, 1, 1, 1, 1])
    encoder = IdentityEncoder()
    torso = MLP([5, 5])
    head = DeterministicHead(input_shape=5, action_shape=1)

    actor = Actor(encoder, torso, head)

    output = actor(input)
    assert output.shape == (1,)

    actor = Actor(encoder, torso, head, create_target=True)

    online_output = actor(input)
    target_output = actor.forward_target(input)
    assert T.equal(online_output, target_output)

    state_shape = actor.numpy().shape
    new_state = np.random.rand(*state_shape)
    actor.set_state(new_state)
    new_output = actor(input)
    assert not T.equal(online_output, new_output)
    np.testing.assert_array_equal(actor.numpy(), new_state)

    online_output = actor(input)
    new_target_output = actor.forward_target(input)
    assert not T.equal(online_output, new_target_output)
    assert T.equal(target_output, new_target_output)

    actor.update_targets()
    update_target_output = actor.forward_target(input)
    assert not T.equal(new_target_output, update_target_output)
    assert not T.equal(online_output, update_target_output)

    actor.assign_targets()
    target_output = actor.forward_target(input)
    assert T.equal(online_output, target_output)
    assert actor.action_distribution(input) is None


@pytest.mark.parametrize("actor_population_size", [1, 2])
@pytest.mark.parametrize("critic_population_size", [1, 2])
def test_actor_critic(actor_population_size, critic_population_size):
    input = T.Tensor([1, 1, 1, 1, 1])
    x_actor = input.repeat(actor_population_size, 1)
    x_critic = input.repeat(critic_population_size, 1)
    encoder_actor = IdentityEncoder()
    encoder_critic = IdentityEncoder()
    torso_actor = MLP([5, 5])
    torso_critic = MLP([5, 5])
    head_critic = ValueHead(input_shape=5)
    head_actor = DeterministicHead(input_shape=5, action_shape=1)

    actor = Actor(encoder_actor, torso_actor, head_actor)
    critic = Critic(encoder_critic, torso_critic, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=actor_population_size,
            critic_population_size=critic_population_size,
            actor_distribution="uniform",
            critic_distribution=None,
        ),
    )
    actor_out = model(x_actor)
    critic_out = model.forward_critics(x_critic)

    actor_state = model.numpy_actors()
    critic_state = model.numpy_critics()

    new_actor_state = np.random.rand(*actor_state.shape)
    new_critic_state = np.random.rand(*critic_state.shape)

    model.set_actors_state(new_actor_state)
    assert T.equal(model.forward_critics(x_critic), critic_out)
    assert not T.equal(model(x_actor), actor_out)

    model.set_critics_state(new_critic_state)
    assert not T.equal(model.forward_critics(x_critic), critic_out)

    old_action = model.predict(input)
    old_value = model.predict_critic(input)
    model.update_global()
    actual_actor_state = model.actor.numpy()
    actual_critic_state = model.critic.numpy()
    expected_actor_state = (
        new_actor_state.squeeze()
        if actor_population_size == 1
        else np.mean([actor.state for actor in model.actors], axis=0)
    )
    expected_critic_state = (
        new_critic_state.squeeze()
        if critic_population_size == 1
        else np.mean([critic.state for critic in model.critics], axis=0)
    )
    np.testing.assert_array_equal(actual_actor_state, expected_actor_state)
    np.testing.assert_array_equal(actual_critic_state, expected_critic_state)
    new_action = model.predict(input)
    new_value = model.predict_critic(input)
    assert not T.equal(old_action, new_action)
    assert not T.equal(old_value, new_value)
    assert model.predict_distribution(input) is None


def test_shared_network():
    # Shared encoder and torso
    encoder = IdentityEncoder()
    torso = MLP([5, 5])
    head_actor = DeterministicHead(input_shape=5, action_shape=1)
    head_critic = ValueHead(input_shape=5)

    actor = Actor(encoder, torso, head_actor)
    critic = Critic(encoder, torso, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=1,
            critic_population_size=1,
            actor_distribution="normal",
            critic_distribution=None,
        ),
    )

    assert model.actor.model.encoder == model.critic.model.encoder
    assert model.actor.model.torso == model.critic.model.torso
    assert model.actor.model.head != model.critic.model.head
    for actor, critic in zip(model.actors, model.critics):
        assert actor.model.encoder == critic.model.encoder
        assert actor.model.torso == critic.model.torso
        assert actor.model.head != critic.model.head

    # Shared encoder
    encoder = IdentityEncoder()
    torso_actor = MLP([5, 5])
    torso_critic = MLP([5, 5])
    head_actor = DeterministicHead(input_shape=5, action_shape=1)
    head_critic = ValueHead(input_shape=5)

    actor = Actor(encoder, torso_actor, head_actor)
    critic = Critic(encoder, torso_critic, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=1,
            critic_population_size=1,
            actor_distribution="normal",
            critic_distribution=None,
        ),
    )

    assert model.actor.model.encoder == model.critic.model.encoder
    assert model.actor.model.torso != model.critic.model.torso
    assert model.actor.model.head != model.critic.model.head
    for actor, critic in zip(model.actors, model.critics):
        assert actor.model.encoder == critic.model.encoder
        assert actor.model.torso != critic.model.torso
        assert actor.model.head != critic.model.head


@pytest.mark.parametrize("actor_population_size", [1, 2])
@pytest.mark.parametrize("critic_population_size", [1, 2])
def test_actor_critic_targets(actor_population_size, critic_population_size):
    input = T.Tensor([1, 1, 1, 1, 1])
    x_actor = input.repeat(actor_population_size, 1)
    x_critic = input.repeat(critic_population_size, 1)
    encoder_actor = IdentityEncoder()
    encoder_critic = IdentityEncoder()
    torso_actor = MLP([5, 5])
    torso_critic = MLP([5, 5])
    head_critic = ValueHead(input_shape=5)
    head_actor = DeterministicHead(input_shape=5, action_shape=1)

    actor = Actor(encoder_actor, torso_actor, head_actor, create_target=True)
    critic = Critic(encoder_critic, torso_critic, head_critic, create_target=True)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=actor_population_size,
            critic_population_size=critic_population_size,
        ),
    )

    actor_out = model(x_actor)
    actor_target_out = model.forward_target_actors(x_actor)
    critic_out = model.forward_critics(x_critic)
    critic_target_out = model.forward_target_critics(x_critic)

    assert T.equal(actor_out, actor_target_out)
    assert T.equal(critic_out, critic_target_out)

    actor_state = model.numpy_actors()
    critic_state = model.numpy_critics()

    new_actor_state = np.random.rand(*actor_state.shape)
    new_critic_state = np.random.rand(*critic_state.shape)

    model.set_actors_state(new_actor_state)
    assert T.equal(model.forward_critics(x_critic), critic_out)
    assert not T.equal(model(x_actor), actor_target_out)
    assert T.equal(model.forward_target_actors(x_actor), actor_target_out)

    model.set_critics_state(new_critic_state)
    assert not T.equal(model.forward_critics(x_critic), critic_target_out)
    assert T.equal(model.forward_target_critics(x_critic), critic_target_out)

    model.update_targets()
    model.assign_targets()
    assert not T.equal(model.forward_target_critics(x_critic), critic_target_out)
    assert T.equal(
        model.forward_target_critics(x_critic), model.forward_critics(x_critic)
    )
    assert not T.equal(model.forward_target_actors(x_actor), actor_target_out)
    assert T.equal(model.forward_target_actors(x_actor), model(x_actor))


def test_population_initialize():
    encoder_actor = IdentityEncoder()
    encoder_critic = IdentityEncoder()
    torso_actor = MLP([5, 5])
    torso_critic = MLP([5, 5])
    head_critic = ValueHead(input_shape=5)
    head_actor = DeterministicHead(input_shape=5, action_shape=1)

    actor = Actor(encoder_actor, torso_actor, head_actor)
    critic = Critic(encoder_critic, torso_critic, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=500, actor_distribution="normal"
        ),
    )

    actor_state = model.numpy_actors()

    np.testing.assert_array_almost_equal(np.std(actor_state), 1, decimal=1.5)


@pytest.mark.parametrize("actor_population_size", [1, 2])
def test_action_distribution(actor_population_size):
    input = T.Tensor([1, 1, 1, 1, 1]).repeat(actor_population_size, 1)
    encoder_actor = IdentityEncoder()
    encoder_critic = IdentityEncoder()
    torso_actor = MLP([5, 5])
    torso_critic = MLP([5, 5])
    head_critic = ValueHead(input_shape=5)
    head_actor = DeterministicHead(input_shape=5, action_shape=1)

    actor = Actor(encoder_actor, torso_actor, head_actor)
    critic = Critic(encoder_critic, torso_critic, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=actor_population_size,
        ),
    )

    distribution = model.action_distribution(input)
    global_dist = model.predict_distribution(T.Tensor([1, 1, 1, 1, 1]))
    assert distribution is None
    assert global_dist is None

    input = T.Tensor([1, 1, 1, 1, 1]).repeat(actor_population_size, 1)
    head_actor = CategoricalHead(input_shape=5, action_size=5)

    actor = Actor(encoder_actor, torso_actor, head_actor)
    critic = Critic(encoder_critic, torso_critic, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=actor_population_size,
        ),
    )

    distribution = model.action_distribution(input)
    global_dist = model.predict_distribution(T.Tensor([1, 1, 1, 1, 1]))
    assert isinstance(distribution, T.distributions.Categorical)
    assert distribution.logits.shape == (actor_population_size, 5)
    assert global_dist.logits.shape == (5,)

    input = T.Tensor([1, 1, 1, 1, 1]).repeat(actor_population_size, 1)
    head_actor = DiagGaussianHead(input_shape=5, action_size=5)

    actor = Actor(encoder_actor, torso_actor, head_actor)
    critic = Critic(encoder_critic, torso_critic, head_critic)

    model = ActorCritic(
        actor,
        critic,
        population_settings=PopulationSettings(
            actor_population_size=actor_population_size,
        ),
    )

    distribution = model.action_distribution(input)
    global_dist = model.predict_distribution(T.Tensor([1, 1, 1, 1, 1]))
    assert isinstance(distribution, T.distributions.Normal)
    assert distribution.loc.shape == (actor_population_size, 5)
    assert distribution.scale.shape == (actor_population_size, 5)
    assert global_dist.loc.shape == (5,)
    assert global_dist.scale.shape == (5,)


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


@pytest.mark.parametrize("model_class", [DummyActor, DummyCritic])
def test_dummy(model_class):
    env = gym.make("CartPole-v0")
    expected_state = env.action_space.sample()
    model = model_class(space=env.action_space, state=expected_state)

    actual_state = model.numpy()
    np.testing.assert_array_equal(actual_state, expected_state)

    expected_state = env.action_space.sample()
    model.set_state(expected_state)

    np.testing.assert_array_equal(model.numpy(), expected_state)

    actual_state = model(env.observation_space.sample())
    print(actual_state, expected_state)
    np.testing.assert_array_almost_equal(actual_state, expected_state)
