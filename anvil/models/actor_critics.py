import copy
from typing import List, Optional, Tuple, Type, Union

import torch as T

from anvil.models.heads import BaseActorHead, BaseCriticHead
from anvil.models.utils import get_device, trainable_variables


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
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.head = head.to(device)
        self.optimizer = optimizer_class(self.parameters(), lr=lr)

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
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.head = head.to(device)
        self.optimizer = optimizer_class(self.parameters(), lr=lr)

    def forward(self, *inputs) -> List[T.Tensor]:
        out = self.encoder(*inputs)
        out = self.torso(out)
        out = self.head(out)
        return out


class ActorCritic(T.nn.Module):
    """
    A basic actor critic network.
    This module allows for a shared architecture, which assumes the actor layers
    should be used as the shared layers if specified. By default, the encoder is
    shared since we assume an encoder is used that does not have trainable variables
    (e.g. FlattenEncoder) so it makes sense to save the memory.

    :param actor: the actor/policy network
    :param critic: the critic network
    :param share_encoder: whether to use a shared encoder for both networks
    :param share_torso: whether to use a shared torso for both networks
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        share_encoder: bool = True,
        share_torso: bool = False,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.share_encoder = share_encoder
        self.share_torso = share_torso
        self.encoder = None  # encoder if shared encoder
        self.encoder_actor = None  # actor encoder if separate encoders but shared torso
        # critic encoder if separate encoders but shared torso
        self.encoder_critic = None
        self.torso = None  # torso if shared torso
        self.torso_actor = None  # actor torso if separate torsos but shared encoder
        self.torso_critic = None  # critic torso if separate torsos but shared encoder
        self.head_actor = None  # actor head if shared layers
        self.head_critic = None  # critic head if share layers
        self.online_variables = None
        self.target_variables = None
        self.polyak_coeff = None

        if share_encoder or share_torso:
            self._build()

    def _build(self) -> None:
        """Define the shared layers of the network"""
        if self.share_encoder:
            self.encoder = self.actor.encoder
        else:
            self.encoder_actor = self.actor.encoder
            self.encoder_critic = self.critic.encoder
        if self.share_torso:
            self.torso = self.actor.torso
        else:
            self.torso_actor = self.actor.torso
            self.torso_critic = self.critic.torso

        self.head_actor = self.actor.head
        self.head_critic = self.critic.head

        self.actor = None
        self.critic = None

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

    def _forward_actor_encoder(self, *inputs) -> T.Tensor:
        """Given a shared architecture, get the actor encoder output"""
        if self.share_encoder:
            return self.encoder(*inputs)
        else:
            return self.encoder_actor(*inputs)

    def _forward_critic_encoder(self, *inputs) -> T.Tensor:
        """Given a shared architecture, get the critic encoder output"""
        if self.share_encoder:
            return self.encoder(*inputs)
        else:
            return self.encoder_critic(*inputs)

    def _forward_actor_torso(self, input: T.Tensor) -> T.Tensor:
        """Given a shared architecture, get the actor torso output"""
        if self.share_torso:
            return self.torso(input)
        else:
            return self.torso_actor(input)

    def _forward_critic_torso(self, input: T.Tensor) -> T.Tensor:
        """Given a shared architecture, get the critic torso output"""
        if self.share_torso:
            return self.torso(input)
        else:
            return self.torso_critic(input)

    def get_action_distribution(
        self, *inputs
    ) -> Optional[T.distributions.Distribution]:
        """Get the action distribution, returns None if deterministic"""
        if not self.share_torso and not self.share_encoder:
            return self.actor.get_action_distribution(*inputs)
        else:
            latent_out = self._forward_actor_torso(self._forward_actor_encoder(*inputs))
            return self.head_actor.get_action_distribution(latent_out)

    def forward_critic(self, *inputs) -> T.Tensor:
        """Run a forward pass to get the critic output"""
        if not self.share_torso and not self.share_encoder:
            return self.critic(*inputs)
        else:
            out = self._forward_critic_encoder(*inputs)
            out = self._forward_critic_torso(out)
            return self.head_critic(out)

    def forward(self, *inputs) -> T.Tensor:
        """The default forward pass retrieves an action prediciton"""
        if not self.share_encoder and not self.share_torso:
            return self.actor(*inputs)
        else:
            out = self._forward_actor_encoder(*inputs)
            out = self._forward_actor_torso(out)
            return self.head_actor(out)


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
        share_encoder: bool = True,
        share_torso: bool = False,
        polyak_coeff: float = 0.995,
    ) -> None:
        super().__init__(actor, critic, share_encoder, share_torso)
        self.polyak_coeff = polyak_coeff

        if not self.share_encoder and not self.share_torso:
            self.target_critic = copy.deepcopy(critic)
            self.online_variables = trainable_variables(self.critic)
        else:
            # assign encoder variables and target layer
            if self.share_encoder:
                target_encoder = copy.deepcopy(self.encoder)
                self.online_variables = trainable_variables(self.encoder)
            else:
                target_encoder = copy.deepcopy(self.encoder_critic)
                self.online_variables = trainable_variables(self.encoder_critic)

            # assign torso variables and target layer
            if self.share_torso:
                target_torso = copy.deepcopy(self.torso)
                self.online_variables += trainable_variables(self.torso)
            else:
                target_torso = copy.deepcopy(self.torso_critic)
                self.online_variables += trainable_variables(self.torso_critic)

            # assign head variables and target layer
            target_head = copy.deepcopy(self.head_critic)
            self.online_variables += trainable_variables(self.head_critic)

            # create target network from above defined target layers
            self.target_critic = T.nn.Sequential(
                target_encoder, target_torso, target_head
            )
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
        share_encoder: bool = False,
        share_torso: bool = False,
        polyak_coeff: float = 0.995,
    ) -> None:
        super().__init__(actor, critic, share_encoder, share_torso)
        self.polyak_coeff = polyak_coeff
        if not self.share_encoder and not self.share_torso:
            self.critic_2 = copy.deepcopy(critic)
            self.target_critic = copy.deepcopy(critic)
            self.target_critic_2 = copy.deepcopy(critic)
            self.online_variables = trainable_variables(self.critic)
            self.online_variables += trainable_variables(self.critic_2)
        else:
            # assign encoder variables and target layer
            if self.share_encoder:
                encoder_2 = copy.deepcopy(self.encoder)
                target_encoder = copy.deepcopy(self.encoder)
                target_encoder_2 = copy.deepcopy(self.encoder)
                self.online_variables = trainable_variables(self.encoder)
            else:
                encoder_2 = copy.deepcopy(self.encoder_critic)
                target_encoder = copy.deepcopy(self.encoder_critic)
                target_encoder_2 = copy.deepcopy(self.encoder_critic)
                self.online_variables = trainable_variables(self.encoder_critic)

            # assign torso variables and target layer
            if self.share_torso:
                torso_2 = copy.deepcopy(self.torso)
                target_torso = copy.deepcopy(self.torso)
                target_torso_2 = copy.deepcopy(self.torso)
                self.online_variables += trainable_variables(self.torso)
            else:
                torso_2 = copy.deepcopy(self.torso_critic)
                target_torso = copy.deepcopy(self.torso_critic)
                target_torso_2 = copy.deepcopy(self.torso_critic)
                self.online_variables += trainable_variables(self.torso_critic)

            # assign head variables and target layer
            head_2 = copy.deepcopy(self.head_critic)
            target_head = copy.deepcopy(self.head_critic)
            target_head_2 = copy.deepcopy(self.head_critic)
            self.online_variables += trainable_variables(self.head_critic)

            # define the other critic networks
            self.critic_2 = T.nn.Sequential(encoder_2, torso_2, head_2)
            self.target_critic = T.nn.Sequential(
                target_encoder, target_torso, target_head
            )
            self.target_critic_2 = T.nn.Sequential(
                target_encoder_2, target_torso_2, target_head_2
            )

        self.online_variables += trainable_variables(self.critic_2)
        self.target_variables = trainable_variables(self.target_critic)
        self.target_variables += trainable_variables(self.target_critic_2)

        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()

    def forward_critic(self, *inputs) -> Tuple[T.Tensor, T.Tensor]:
        """Run a forward pass to get the critic outputs"""
        if not self.share_torso and not self.share_encoder:
            return self.critic(*inputs), self.critic_2(*inputs)
        else:
            out = self._forward_critic_encoder(*inputs)
            out = self._forward_critic_torso(out)
            return self.head_critic(out), self.critic_2(*inputs)
