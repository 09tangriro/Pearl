from typing import Optional, Type

import torch as T
from gym import spaces

from anvil_rl.common.type_aliases import Tensor
from anvil_rl.common.utils import numpy_to_torch
from anvil_rl.models.utils import concat_obs_actions, is_image_space


class IdentityEncoder(T.nn.Module):
    """This encoder passes the input through unchanged."""

    def __init__(self):
        super().__init__()

    def forward(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        # Some algorithms use both the observations and actions as input (e.g. DDPG for conitnuous Q function)
        observations = concat_obs_actions(observations, actions)
        return numpy_to_torch(observations)


class FlattenEncoder(T.nn.Module):
    """This encoder flattens the input."""

    def __init__(self):
        super().__init__()
        self.flatten = T.nn.Flatten()

    def forward(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        # Some algorithms use both the observations and actions as input (e.g. DDPG for conitnuous Q function)
        observations = concat_obs_actions(observations, actions)
        # Make sure observations is a torch tensor, get error if numpy for some reason??
        observations = numpy_to_torch(observations)
        return self.flatten(observations)


class MLPEncoder(T.nn.Module):
    """This is a single layer MLP encoder"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = T.nn.Linear(input_size, output_size)

    def forward(
        self, observations: Tensor, actions: Optional[Tensor] = None
    ) -> T.Tensor:
        observations = concat_obs_actions(observations, actions)
        return self.model(observations)


class CNNEncoder(T.nn.Module):
    """
    CNN from DQN nature paper:
    Mnih, Volodymyr, et al.
    "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param output_size: number neurons in the last layer.
    :param activation_fn: the activation function after each layer
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        output_size: int = 512,
        activation_fn: Type[T.nn.Module] = T.nn.ReLU,
    ):
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN " f"only with images not with {observation_space}"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = T.nn.Sequential(
            T.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            activation_fn(),
            T.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            activation_fn(),
            T.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            activation_fn(),
            T.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with T.no_grad():
            n_flatten = self.cnn(
                T.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = T.nn.Sequential(T.nn.Linear(n_flatten, output_size), T.nn.ReLU())

    def forward(self, observations: Tensor) -> T.Tensor:
        return self.linear(self.cnn(observations))
