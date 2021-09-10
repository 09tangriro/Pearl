from typing import Optional, Tuple, Type, Union

import torch as T

from anvil.models.torsos import MLP
from anvil.models.utils import NetworkType


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
