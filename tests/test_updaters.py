import pytest
import torch as T

from anvil.models import Actor, ActorCritic, Critic
from anvil.models.encoders import IdentityEncoder
from anvil.models.heads import DiagGaussianPolicyHead, ValueHead
from anvil.models.torsos import MLP
from anvil.updaters.actors import PolicyGradient
from anvil.updaters.utils import (
    sample_forward_kl_divergence,
    sample_reverse_kl_divergence,
)

encoder_critic = IdentityEncoder()
encoder_actor = IdentityEncoder()
torso_critic = MLP([2, 2])
torso_actor = MLP([2, 2])
head_actor = DiagGaussianPolicyHead(input_shape=2, action_size=1)
head_critic = ValueHead(input_shape=2, activation_fn=None)
actor = Actor(encoder=encoder_actor, torso=torso_actor, head=head_actor)
critic = Critic(encoder=encoder_critic, torso=torso_critic, head=head_critic)
critic_shared_encoder = Critic(
    encoder=encoder_actor, torso=torso_critic, head=head_critic
)
critic_shared = Critic(encoder=encoder_actor, torso=torso_actor, head=head_critic)
actor_critic = ActorCritic(actor=actor, critic=critic)
actor_critic_shared_encoder = ActorCritic(actor=actor, critic=critic_shared_encoder)
actor_critic_shared = ActorCritic(actor=actor, critic=critic_shared)


@pytest.mark.parametrize(
    "kl_divergence", [sample_forward_kl_divergence, sample_reverse_kl_divergence]
)
def test_kl_divergence(kl_divergence):
    # KL Div = E(unbiased approx KL Div)
    T.manual_seed(8)
    num_samples = int(2e6)
    dist1 = T.distributions.Normal(0, 1)
    dist2 = T.distributions.Normal(1, 2)

    samples = dist2.sample((num_samples,))

    log_probs1 = dist1.log_prob(samples)
    log_probs2 = dist2.log_prob(samples)

    if kl_divergence == sample_forward_kl_divergence:
        full_kl_div = T.distributions.kl_divergence(dist1, dist2)
    else:
        full_kl_div = T.distributions.kl_divergence(dist2, dist1)

    approx_kl_div = kl_divergence(log_probs1.exp(), log_probs2.exp()).mean()

    assert T.allclose(full_kl_div, approx_kl_div, rtol=5e-3)


@pytest.mark.parametrize(
    "model", [actor, actor_critic, actor_critic_shared_encoder, actor_critic_shared]
)
def test_policy_gradient(model):
    observation = T.rand(2)
    out_before = model(observation)
    if model != actor:
        with T.no_grad():
            critic_before = model.forward_critic(observation)

    updater = PolicyGradient(max_grad=0.5)
    if model == actor_critic:
        updater(
            model=model.actor,
            observations=observation,
            actions=T.rand(1),
            advantages=T.rand(1),
        )
    else:
        updater(
            model=model,
            observations=observation,
            actions=T.rand(1),
            advantages=T.rand(1),
        )

    out_after = model(observation)
    if model != actor:
        with T.no_grad():
            critic_after = model.forward_critic(observation)

    assert out_after != out_before
    if model == actor_critic or model == actor_critic_shared_encoder:
        assert critic_after == critic_before
    if model == actor_critic_shared:
        assert critic_after != critic_before