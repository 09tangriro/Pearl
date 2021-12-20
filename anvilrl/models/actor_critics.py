import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch as T
from gym.spaces import Box, Space
from gym.vector import VectorEnv

from anvilrl.common.type_aliases import Tensor
from anvilrl.common.utils import (
    get_device,
    get_space_range,
    get_space_shape,
    numpy_to_torch,
    torch_to_numpy,
)
from anvilrl.models.heads import BaseActorHead, BaseCriticHead
from anvilrl.models.utils import trainable_parameters


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
        self.device = get_device(device)
        self.encoder = encoder.to(self.device)
        self.torso = torso.to(self.device)
        self.head = head.to(self.device)

    def get_action_distribution(
        self, observations: Tensor
    ) -> Optional[T.distributions.Distribution]:
        """Get the action distribution, returns None if deterministic"""
        latent_out = self.torso(self.encoder(observations))
        return self.head.get_action_distribution(latent_out)

    def forward(self, observations: Tensor) -> T.Tensor:
        out = self.encoder(observations)
        out = self.torso(out)
        out = self.head(out)
        return out


class EpsilonGreedyActor(Actor):
    """
    Epsilon greedy strategy used in DQN
    Epsilon represents the probability of choosing a random action in the environment

    :param critic_encoder: encoder of the critic network
    :param critic_torso: torso of the critic network
    :param critic_head: head of the critic network
    :param start_epsilon: epsilon start value, generally set high to promp random exploration
    :param epsilon_decay: epsilon decay value, epsilon = epsilon * epsilon_decay
    :param min_epsilon: the minimum epsilon value allowed
    """

    def __init__(
        self,
        critic_encoder: T.nn.Module,
        critic_torso: T.nn.Module,
        critic_head: BaseCriticHead,
        start_epsilon: float = 1,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__(critic_encoder, critic_torso, critic_head, device=device)
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def update_epsilon(self) -> None:
        """Update epsilon via epsilon decay"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def forward(self, observations: Tensor) -> T.Tensor:
        q_values = super().forward(observations)
        action_size = q_values.shape[-1]
        trigger = T.rand(1).item()

        if trigger <= self.epsilon:
            actions = T.randint(low=0, high=action_size, size=q_values.shape[:-1])
        else:
            _, actions = T.max(q_values, dim=-1)

        self.update_epsilon()
        return actions


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

    def forward(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> List[T.Tensor]:
        out = self.encoder(observations, actions)
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
        head_actor = DiagGaussianHead(input_shape=2, action_size=1)
        head_critic = ValueHead(input_shape=2, activation_fn=None)
        actor = Actor(encoder=encoder_actor, torso=torso_actor, head=head_actor)
        critic = Critic(encoder=encoder_actor, torso=torso_critic, head=head_critic)
        actor_critic = ActorCritic(actor=actor, critic=critic)

    Shared architecture:
        encoder = IdentityEncoder()
        torso = MLP([2, 2])
        head_actor = DiagGaussianHead(input_shape=2, action_size=1)
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
        self.online_parameters = None
        self.target_parameters = None
        self.polyak_coeff = None

    def assign_targets(self) -> None:
        """Assign the target parameters"""
        for online, target in zip(self.online_parameters, self.target_parameters):
            target.data.copy_(online.data)

    def update_targets(self) -> None:
        """Update the target parameters"""
        # target_params = polyak_coeff * target_params + (1 - polyak_coeff) * online_params
        with T.no_grad():
            for online, target in zip(self.online_parameters, self.target_parameters):
                target.data.mul_(self.polyak_coeff)
                target.data.add_((1 - self.polyak_coeff) * online.data)

    def get_action_distribution(
        self, observations: Tensor
    ) -> Optional[T.distributions.Distribution]:
        """Get the action distribution, returns None if deterministic"""
        return self.actor.get_action_distribution(observations)

    def forward_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Run a forward pass to get the critic output"""
        return self.critic(observations, actions)

    def forward(self, observations: Tensor) -> T.Tensor:
        """The default forward pass retrieves an action prediciton"""
        return self.actor(observations)


class ActorCriticWithCriticTarget(ActorCritic):
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
        polyak_coeff: float = 0.995,
    ) -> None:
        super().__init__(actor, critic)
        self.polyak_coeff = polyak_coeff

        self.target_critic = copy.deepcopy(critic)
        self.online_parameters = trainable_parameters(
            self.critic
        ) + trainable_parameters(self.actor)
        self.target_parameters = trainable_parameters(self.target_critic)

        for target in self.target_parameters:
            target.requires_grad = False
        self.assign_targets()

    def forward_target_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Run a forward pass to get the target critic output"""
        return self.target_critic(observations, actions)


class ActorCriticWithTargets(ActorCritic):
    """
    An actor critic with target critic and actor networks updated via polyak averaging

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

        self.target_critic = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.online_parameters = trainable_parameters(
            self.critic
        ) + trainable_parameters(self.actor)
        self.target_parameters = trainable_parameters(
            self.target_critic
        ) + trainable_parameters(self.target_actor)

        for target in self.target_parameters:
            target.requires_grad = False
        self.assign_targets()

    def forward_target_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Run a forward pass to get the target critic output"""
        return self.target_critic(observations, actions)

    def forward_target_actor(self, observations: Tensor) -> T.Tensor:
        """Run a forward pass to get the target actor output"""
        return self.target_actor(observations)


class TwinActorCritic(ActorCritic):
    """
    The TwinActorCritic actor critic model with 2 critic networks each with their own target

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
        self.critic2 = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.target_critic2 = copy.deepcopy(critic)
        self.online_parameters = trainable_parameters(self.critic)
        self.online_parameters += trainable_parameters(self.critic2)
        self.online_parameters += trainable_parameters(self.actor)
        self.target_parameters = trainable_parameters(self.target_critic)
        self.target_parameters += trainable_parameters(self.target_critic2)
        self.target_parameters += trainable_parameters(self.target_actor)

        for target in self.target_parameters:
            target.requires_grad = False
        self.assign_targets()

    def forward_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> Tuple[T.Tensor, T.Tensor]:
        """Run a forward pass to get the critic outputs"""
        return self.critic(observations, actions), self.critic2(observations, actions)

    def forward_target_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> Tuple[T.Tensor, T.Tensor]:
        """Run a forward pass to get the target critics outputs"""
        return self.target_critic(observations, actions), self.target_critic2(
            observations, actions
        )

    def forward_target_actor(self, observations: Tensor) -> T.Tensor:
        """Run a forward pass to get the target actor output"""
        return self.target_actor(observations)


class Individual(T.nn.Module):
    """
    An individual in the population without a nerual network is not needed.

    :param space: the individual space
    :param state: optional starting state of the individual
    """

    def __init__(self, space: Space, state: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.space = space
        self.space_shape = get_space_shape(self.space)
        self.space_range = get_space_range(self.space)
        self.state = state if state is not None else space.sample()

    def set_state(self, state: np.ndarray) -> "Individual":
        """Set the state of the individual"""
        self.state = state
        return self

    def numpy(self) -> np.ndarray:
        """Get the numpy representation of the individual"""
        return self.state

    def forward(self, observation: Tensor) -> np.ndarray:
        return self.state


class DeepIndividual(Actor):
    """
    An individual in the population with a neural network structure.

    :param encoder: the encoder network
    :param torso: the torso network
    :param head: the head network
    :param space: the individual space
    :param device: the device to use
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        head: BaseActorHead,
        space: Optional[Space] = None,
        device: Union[T.device, str] = "auto",
    ) -> None:
        super().__init__(encoder=encoder, torso=torso, head=head, device=device)
        self.state_info = {}
        self.make_state_info()
        self.state = np.concatenate(
            [torch_to_numpy(d.flatten()) for d in self.state_dict().values()]
        )
        self.space = (
            space
            if space is not None
            else Box(low=-1e6, high=1e6, shape=self.state.shape)
        )
        self.space_shape = get_space_shape(self.space)
        self.space_range = get_space_range(self.space)

    def make_state_info(self) -> None:
        """Make the state info dictionary"""
        self.state_info = {}
        start_idx = 0
        for k, v in self.state_dict().items():
            self.state_info[k] = (v.shape, (start_idx, start_idx + v.numel()))
            start_idx += v.numel()

    def set_state(self, state: np.ndarray) -> "DeepIndividual":
        """Set the state of the individual"""
        self.state = state
        state = numpy_to_torch(state, device=self.device)
        self.load_state_dict(self.state_dict())
        state_dict = {
            k: state[v[1][0] : v[1][1]].reshape(v[0])
            for k, v in zip(self.state_info.keys(), self.state_info.values())
        }
        self.load_state_dict(state_dict)
        return self

    def numpy(self) -> np.ndarray:
        """Get the numpy representation of the individual"""
        return self.state

    def forward(self, observation: Tensor) -> np.ndarray:
        out = super().forward(observation)
        return torch_to_numpy(out)
