import copy
from typing import List, Optional, Tuple, Type, Union

import torch as T

from anvil.common.utils import get_device
from anvil.models.heads import BaseActorHead, BaseCriticHead
from anvil.models.utils import trainable_variables


class Actor(T.nn.Module):
    """
    The actor network which determines what actions to take in the environment.

    :param encoder: the encoder network
    :param torso: the torso network
    :param head: the head network
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        head: BaseActorHead,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.head = head.to(device)

    def get_action_distribution(
        self, *inputs
    ) -> Optional[T.distributions.Distribution]:
        """Get the action distribution, returns None if deterministic"""
        latent_out = self.torso(self.encoder(*inputs))
        return self.head.get_action_distribution(latent_out)

    def forward(self, *inputs) -> List[T.Tensor]:
        out = self.encoder(*inputs)
        out = self.torso(out)
        out = self.head(out)
        return out


class Critic(T.nn.Module):
    """
    The critic network which approximates the Q or Value functions.

    :param encoder: the encoder network
    :param torso: the torso network
    :param head: the head network
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        head: BaseCriticHead,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.head = head.to(device)

    def forward(self, *inputs) -> List[T.Tensor]:
        out = self.encoder(*inputs)
        out = self.torso(out)
        out = self.head(out)
        return out


class ActorCritic(T.nn.Module):
    """
    A basic actor critic network.
    This module is designed flexibly to allow for shared or separate
    network architectures. To define shared layers, simply have the
    actor and critic embedded networks use the same encoder/torso/head
    in memory.

    Separate architecture:
        encoder_critic = IdentityEncoder()
        encoder_actor = IdentityEncoder()
        torso_critic = MLP([2, 2])
        torso_actor = MLP([2, 2])
        head_actor = DiagGaussianPolicyHead(input_shape=2, action_size=1)
        head_critic = ValueHead(input_shape=2, activation_fn=None)
        actor = Actor(encoder=encoder_actor, torso=torso_actor, head=head_actor)
        critic = Critic(encoder=encoder_actor, torso=torso_critic, head=head_critic)
        actor_critic = ActorCritic(actor=actor, critic=critic)

    Shared architecture:
        encoder = IdentityEncoder()
        torso = MLP([2, 2])
        head_actor = DiagGaussianPolicyHead(input_shape=2, action_size=1)
        head_critic = ValueHead(input_shape=2, activation_fn=None)
        actor = Actor(encoder=encoder, torso=torso, head=head_actor)
        critic = Critic(encoder=encoder, torso=torso, head=head_critic)
        actor_critic = ActorCritic(actor=actor, critic=critic)

    :param actor: the actor/policy network
    :param critic: the critic network
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.online_variables = None
        self.target_variables = None
        self.polyak_coeff = None

    def assign_targets(self) -> None:
        """Assign the target variables"""
        for online, target in zip(self.online_variables, self.target_variables):
            target.data.copy_(online.data)

    def update_targets(self) -> None:
        """Update the target variables"""
        # target_params = polyak_coeff * target_params + (1 - polyak_coeff) * online_params
        with T.no_grad():
            for online, target in zip(self.online_variables, self.target_variables):
                target.data.mul_(self.polyak_coeff)
                target.data.add_((1 - self.polyak_coeff) * online.data)

    def get_action_distribution(
        self, *inputs
    ) -> Optional[T.distributions.Distribution]:
        """Get the action distribution, returns None if deterministic"""
        return self.actor.get_action_distribution(*inputs)

    def forward_critic(self, *inputs) -> T.Tensor:
        """Run a forward pass to get the critic output"""
        return self.critic(*inputs)

    def forward(self, *inputs) -> T.Tensor:
        """The default forward pass retrieves an action prediciton"""
        return self.actor(*inputs)


class ActorCriticWithTarget(ActorCritic):
    """
    An actor critic with target critic network updated via polyak averaging

    :param actor: the actor/policy network
    :param critic: the critic network
    :param share_encoder: whether to use a shared encoder for both networks
    :param share_torso: whether to use a shared torso for both networks
    :param polyak_coeff: the polyak update coefficient
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        polyak_coeff: float = 0.995,
    ) -> None:
        super().__init__(actor, critic)
        self.polyak_coeff = polyak_coeff

        self.target_critic = copy.deepcopy(critic)
        self.online_variables = trainable_variables(self.critic)
        self.target_variables = trainable_variables(self.target_critic)

        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def forward_target(self, *inputs) -> T.Tensor:
        """Run a forward pass to get the target critic output"""
        return self.target_critic(*inputs)


class TD3ActorCritic(ActorCritic):
    """
    The TD3 actor critic model with 2 critic networks each with their own target

    :param actor: the actor/policy network
    :param critic: the critic network
    :param polyak_coeff: the polyak update coefficient
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        polyak_coeff: float = 0.995,
    ) -> None:
        super().__init__(actor, critic)
        self.polyak_coeff = polyak_coeff
        self.critic_2 = copy.deepcopy(critic)
        self.target_critic = copy.deepcopy(critic)
        self.target_critic_2 = copy.deepcopy(critic)
        self.online_variables = trainable_variables(self.critic)
        self.online_variables += trainable_variables(self.critic_2)
        self.target_variables = trainable_variables(self.target_critic)
        self.target_variables += trainable_variables(self.target_critic_2)

        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def forward_critic(self, *inputs) -> Tuple[T.Tensor, T.Tensor]:
        """Run a forward pass to get the critic outputs"""
        return self.critic(*inputs), self.critic_2(*inputs)
