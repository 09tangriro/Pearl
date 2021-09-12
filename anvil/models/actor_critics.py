import copy
from typing import List, Type, Union

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

    def get_action_distribution(self, *inputs) -> T.distributions.Distribution:
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
    A basic actor critic network

    :param actor: the actor/policy network
    :param critic: the critic network
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
        self.encoder = None
        self.encoder_actor = None
        self.encoder_critic = None
        self.torso = None
        self.torso_actor = None
        self.torso_critic = None
        self.head = None
        self.head_actor = None
        self.head_critic = None
        self.online_variables = None
        self.target_variables = None

        if share_encoder or share_torso:
            self._build()

    def _build(self) -> None:
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

    def forward_critic(self, *inputs) -> T.Tensor:
        if not self.share_torso and not self.share_encoder:
            return self.critic(*inputs)
        else:
            if self.share_encoder:
                out = self.encoder(*inputs)
            else:
                out = self.encoder_critic(*inputs)
            if self.share_torso:
                out = self.torso(out)
            else:
                out = self.torso_critic(out)
            return self.head_critic(out)

    def forward(self, *inputs) -> T.Tensor:
        if not self.share_encoder and not self.share_torso:
            return self.actor(*inputs)
        else:
            if self.share_encoder:
                out = self.encoder(*inputs)
            else:
                out = self.encoder_actor(*inputs)
            if self.share_torso:
                out = self.torso(out)
            else:
                out = self.torso_actor(out)
            return self.head_actor(out)


class ActorCriticWithTarget(ActorCritic):
    """
    An actor critic with target critic network updated via polyak averaging

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
        self.target_critic = copy.deepcopy(critic)

        self.online_variables = trainable_variables(self.critic)
        self.target_variables = trainable_variables(self.target_critic)

        for target in self.target_variables:
            target.requires_grad = False
        self.assign_targets()


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
