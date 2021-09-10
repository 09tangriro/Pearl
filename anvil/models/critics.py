from typing import List, Optional, Tuple, Type, Union

import torch as T

from anvil.models.networks import MLP
from anvil.models.utils import NetworkType, get_device


class BaseHead(T.nn.Module):
    """The base class for the head network"""

    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, input: T.Tensor) -> T.Tensor:
        return self.model(input)


class ValueHead(BaseHead):
    """
    Use this head if you want to model the value function or the continuous Q function.

    :param input_shape: the input shape to the head network, can be the tuple shape or simplified integer input size
    :param activation_fn: the activation function after each layer
    :param network_type: the type of network used
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        network_type: str = "mlp",
        activation_fn: Optional[Type[T.nn.Module]] = T.nn.ReLU,
    ):
        super().__init__()
        network_type = NetworkType(network_type.lower())
        if network_type == NetworkType.MLP:
            if isinstance(input_shape, tuple):
                input_shape = input_shape[0]
            self.model = MLP(layer_sizes=[input_shape, 1], activation_fn=activation_fn)
        else:
            raise NotImplementedError(
                "That type of network hasn't been implemented yet :("
            )


# The continuous Q function has the same structure as
# the value function so we can use the same object.
ContinuousQHead = ValueHead


class DiscreteQHead(BaseHead):
    """
    Use this head if you want to model the discrete Q function.

    :param input_shape: the input shape to the head network, can be the tuple shape or simplified integer input size
    :param output_shape: the output shape of the network, can be the tuple shape or simplified integer output size
    :param activation_fn: the activation function after each layer
    :param network_type: the type of network used
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        output_shape: Union[int, Tuple[int]],
        network_type: str = "mlp",
        activation_fn: Optional[Type[T.nn.Module]] = T.nn.ReLU,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        network_type = NetworkType(network_type.lower())
        if network_type == NetworkType.MLP:
            if isinstance(input_shape, tuple):
                input_shape = input_shape[0]
            if isinstance(output_shape, tuple):
                output_shape = output_shape[0]
            self.model = MLP(
                layer_sizes=[input_shape, output_shape], activation_fn=activation_fn
            )
        else:
            raise NotImplementedError(
                "That type of network hasn't been implemented yet :("
            )


class Critic(T.nn.Module):
    """
    The critic network which approximates the Q or Value functions. Define the heads as a
    list to allow for potential multiple head output networks.

    :param encoder: the encoder network
    :param torso: the torso network
    :param heads: a list of network heads
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        heads: List[T.nn.Module],
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.heads = [head.to(device) for head in heads]

    def forward(self, input: T.Tensor) -> List[T.Tensor]:
        out = self.encoder(input)
        out = self.torso(out)
        out = [head(out) for head in self.heads]
        return out
