import copy
from typing import List, Optional, Union

import numpy as np
import torch as T
from gym.spaces import Box, Discrete, MultiDiscrete, Space

from anvilrl.common.enumerations import Distribution
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
from anvilrl.settings import PopulationSettings


class Model(T.nn.Module):
    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        head: Union[BaseActorHead, BaseCriticHead],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def forward(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        out = self.encoder(observations, actions)
        out = self.torso(out)
        return self.head(out)


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
        create_target: bool = False,
        polyak_coeff: float = 0.995,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        self.polyak_coeff = polyak_coeff
        self.device = get_device(device)
        self.model = Model(encoder, torso, head).to(self.device)
        self.state_info = {}
        self.make_state_info()
        self.state = np.concatenate(
            [torch_to_numpy(d.flatten()) for d in self.model.state_dict().values()]
        )
        self.space = Box(low=-1e6, high=1e6, shape=self.state.shape)
        self.space_shape = get_space_shape(self.space)
        self.space_range = get_space_range(self.space)

        # Create the target network
        self.target = None
        self.online_parameters = None
        self.target_parameters = None
        if create_target:
            self.online_parameters = trainable_parameters(self.model)
            self.target = copy.deepcopy(self.model)
            self.target_parameters = trainable_parameters(self.target)
            for target in self.target_parameters:
                target.requires_grad = False
            self.assign_targets()

    def make_state_info(self) -> None:
        """Make the state info dictionary"""
        start_idx = 0
        for k, v in self.model.state_dict().items():
            self.state_info[k] = (v.shape, (start_idx, start_idx + v.numel()))
            start_idx += v.numel()

    def set_state(self, state: np.ndarray) -> "Actor":
        """Set the state of the individual"""
        self.state = state
        state = numpy_to_torch(state, device=self.device)
        state_dict = {
            k: state[v[1][0] : v[1][1]].reshape(v[0])
            for k, v in zip(self.state_info.keys(), self.state_info.values())
        }
        self.model.load_state_dict(state_dict)
        return self

    def numpy(self) -> np.ndarray:
        """Get the numpy representation of the individual"""
        return self.state

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
        latent_out = self.model.torso(self.model.encoder(observations))
        return self.model.head.get_action_distribution(latent_out)

    def forward_target(self, observations: Tensor) -> T.Tensor:
        return self.target(observations)

    def forward(self, observations: Tensor) -> T.Tensor:
        return self.model(observations)


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
        create_target: bool = False,
        polyak_coeff: float = 0.995,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        self.polyak_coeff = polyak_coeff
        self.device = get_device(device)
        self.model = Model(encoder, torso, head).to(self.device)
        self.state_info = {}
        self.make_state_info()
        self.state = np.concatenate(
            [torch_to_numpy(d.flatten()) for d in self.model.state_dict().values()]
        )
        self.space = Box(low=-1e6, high=1e6, shape=self.state.shape)
        self.space_shape = get_space_shape(self.space)
        self.space_range = get_space_range(self.space)

        # Create the target network
        self.target = None
        self.online_parameters = None
        self.target_parameters = None
        if create_target:
            self.online_parameters = trainable_parameters(self.model)
            self.target = copy.deepcopy(self.model)
            self.target_parameters = trainable_parameters(self.target)
            for target in self.target_parameters:
                target.requires_grad = False
            self.assign_targets()

    def make_state_info(self) -> None:
        """Make the state info dictionary"""
        start_idx = 0
        for k, v in self.model.state_dict().items():
            self.state_info[k] = (v.shape, (start_idx, start_idx + v.numel()))
            start_idx += v.numel()

    def set_state(self, state: np.ndarray) -> "Actor":
        """Set the state of the individual"""
        self.state = state
        state = numpy_to_torch(state, device=self.device)
        state_dict = {
            k: state[v[1][0] : v[1][1]].reshape(v[0])
            for k, v in zip(self.state_info.keys(), self.state_info.values())
        }
        self.model.load_state_dict(state_dict)
        return self

    def numpy(self) -> np.ndarray:
        """Get the numpy representation of the individual"""
        return self.state

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

    def forward_target(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        return self.target(observations, actions)

    def forward(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        return self.model(observations, actions)


class ActorCritic(T.nn.Module):
    """
    A basic actor critic network.
    This module is designed flexibly to allow for:
        1. Shared or separate network architectures.
        2. Multiple actors and/or critics.

    To define shared layers, simply have the
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
        population_settings: PopulationSettings = PopulationSettings(),
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.num_actors = population_settings.actor_population_size
        self.num_critics = population_settings.critic_population_size
        actor_dist = population_settings.actor_distribution
        critic_dist = population_settings.critic_distribution
        actor_dist = (
            Distribution(actor_dist.lower())
            if isinstance(actor_dist, str)
            else actor_dist
        )
        critic_dist = (
            Distribution(critic_dist.lower())
            if isinstance(critic_dist, str)
            else critic_dist
        )
        self.mean_actor = None
        self.normal_dist_actor = None
        self.actors = self.initialize_population(
            model=actor,
            population_size=self.num_actors,
            population_distribution=actor_dist,
            population_std=population_settings.actor_std,
        )
        self.mean_critic = None
        self.normal_dist_critic = None
        self.critics = self.initialize_population(
            model=critic,
            population_size=self.num_critics,
            population_distribution=critic_dist,
            population_std=population_settings.critic_std,
        )
        self.assign_targets()

    def initialize_population(
        self,
        model: Union[Actor, Critic],
        population_size: int,
        population_distribution: Distribution,
        population_std: Union[float, np.ndarray] = 1,
    ) -> List[Union[Actor, Critic]]:
        """
        Initialize the population of networks.
        """
        if population_distribution == Distribution.UNIFORM:
            population = np.random.uniform(
                model.space_range[0],
                model.space_range[1],
                (population_size, *model.space_shape),
            )
        elif population_distribution == Distribution.NORMAL:
            mean = (model.numpy()).astype(np.float32)
            normal_dist = np.random.randn(population_size, *model.space_shape)
            population = mean + (population_std * normal_dist)
            if isinstance(model, Actor):
                self.mean_actor = mean
                self.normal_dist_actor = normal_dist
            elif isinstance(model, Critic):
                self.mean_critic = mean
                self.normal_dist_critic = normal_dist
        else:
            raise ValueError(
                f"The population initialization strategy {population_distribution} is not supported"
            )

        # Discretize and clip population as needed
        if isinstance(model.space, (Discrete, MultiDiscrete)):
            population = np.round(population).astype(np.int32)
        self.population = np.clip(
            population, model.space_range[0], model.space_range[1]
        )

        return [copy.deepcopy(model).set_state(ind) for ind in self.population]

    def numpy_actor(self) -> np.ndarray:
        """
        Get the numpy representation of the actor population.
        """
        return np.array([ind.numpy() for ind in self.actors]).squeeze()

    def numpy_critic(self) -> np.ndarray:
        """
        Get the numpy representation of the critic population.
        """
        return np.array([ind.numpy() for ind in self.critics]).squeeze()

    def set_actor_state(self, state: np.ndarray) -> "ActorCritic":
        """Set the state of the actor"""
        state = state[np.newaxis] if state.ndim == 1 else state
        [actor.set_state(s) for s, actor in zip(state, self.actors)]
        return self

    def set_critic_state(self, state: np.ndarray) -> "ActorCritic":
        """Set the state of the critic"""
        state = state[np.newaxis] if state.ndim == 1 else state
        [critic.set_state(s) for s, critic in zip(state, self.critics)]
        return self

    def assign_targets(self) -> None:
        """Assign the target parameters"""
        if self.actor.target is not None:
            [actor.assign_targets() for actor in self.actors]
        if self.critic.target is not None:
            [critic.assign_targets() for critic in self.critics]

    def update_targets(self) -> None:
        """Update the target parameters"""
        # target_params = polyak_coeff * target_params + (1 - polyak_coeff) * online_params
        if self.actor.target is not None:
            [actor.update_targets() for actor in self.actors]
        if self.critic.target is not None:
            [critic.update_targets() for critic in self.critics]

    def get_action_distribution(
        self, observations: Tensor
    ) -> Optional[List[T.distributions.Distribution]]:
        """Get the action distribution, returns None if deterministic"""
        return [actor.get_action_distribution(observations) for actor in self.actors]

    def forward_target_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Forward the target critic"""
        return T.Tensor(
            [critic.forward_target(observations, actions) for critic in self.critics]
        ).squeeze()

    def forward_target_actor(self, observations: Tensor) -> T.Tensor:
        """Get the target actor output"""
        return T.Tensor(
            [actor.forward_target(observations) for actor in self.actors]
        ).squeeze()

    def forward_critic(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        """Run a forward pass to get the critic output"""
        return T.Tensor(
            [critic(observations, actions) for critic in self.critics]
        ).squeeze()

    def forward(self, observations: Tensor) -> T.Tensor:
        """The default forward pass retrieves an action prediciton"""
        return T.Tensor([actor(observations) for actor in self.actors]).squeeze()


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
