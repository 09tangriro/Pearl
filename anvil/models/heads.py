from typing import Optional, Tuple, Type, Union

import torch as T

from anvil.models.torsos import MLP
from anvil.models.utils import NetworkType, get_mlp_size


class DeterministicBaseHead(T.nn.Module):
    """The base class for the head network"""

    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, input: T.Tensor) -> T.Tensor:
        return self.model(input)


class ValueHead(DeterministicBaseHead):
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
            input_size = get_mlp_size(input_shape)
            self.model = MLP(layer_sizes=[input_size, 1], activation_fn=activation_fn)
        else:
            raise NotImplementedError(f"{network_type} hasn't been implemented yet :(")


# The continuous Q function has the same structure as
# the value function so we can use the same object.
ContinuousQHead = ValueHead


class DiscreteQHead(DeterministicBaseHead):
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
        network_type = NetworkType(network_type.lower())
        if network_type == NetworkType.MLP:
            input_size = get_mlp_size(input_shape)
            output_size = get_mlp_size(output_shape)
            self.model = MLP(
                layer_sizes=[input_size, output_size], activation_fn=activation_fn
            )
        else:
            raise NotImplementedError(f"{network_type} hasn't been implemented yet :(")


class DeterministicPolicyHead(DeterministicBaseHead):
    """
    Use this head if you want a deterministic actor.

    :param input_shape: the input shape to the head network, can be the tuple shape or simplified integer input size
    :param action_shape: the output shape of the network, can be the tuple shape or simplified integer output size
    :param network_type: the type of network used
    :param activation_fn: the activation function after each layer
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        action_shape: Union[int, Tuple[int]],
        network_type: str = "mlp",
        activation_fn: Optional[Type[T.nn.Module]] = T.nn.ReLU,
    ):
        super().__init__()
        network_type = NetworkType(network_type.lower())
        if network_type == NetworkType.MLP:
            input_size = get_mlp_size(input_shape)
            action_size = get_mlp_size(action_shape)
            self.model = MLP(
                layer_sizes=[input_size, action_size], activation_fn=activation_fn
            )
        else:
            raise NotImplementedError(f"{network_type} hasn't been implemented yet :(")


class DiagGaussianPolicyHead(T.nn.Module):
    """
    Use this head if you want a policy obeying a diagonal gaussian distribution.

    :param input_shape: the input shape to the head network, can be the tuple shape or simplified integer input size
    :param action_size: the dimension of the action vector
    :param mean_network_type: network type of the network calculating the mean
    :param mean_activation: activation function of the output of the mean network
    :param log_std_network_type: network type of the network calculating the log std
    :param log_std_activation: activation function of the output of the log std network
    :param log_std_init: if a parameter is used for log std, what initial value should it have
    :param min_log_std: minimum log std of the action distribution
    :param max_log_std: maximum log std of the action distribution
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        action_size: int,
        mean_network_type: str = "mlp",
        mean_activation: Type[T.nn.Module] = T.nn.Tanh,
        log_std_network_type: str = "parameter",
        log_std_activation: Type[T.nn.Module] = T.nn.Softplus,
        log_std_init: float = 0.0,
        min_log_std: Optional[float] = None,
        max_log_std: Optional[float] = None,
    ):
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        mean_network_type = NetworkType(mean_network_type.lower())
        log_std_network_type = NetworkType(log_std_network_type.lower())

        if mean_network_type == NetworkType.MLP:
            input_size = get_mlp_size(input_shape)
            self.mean_network = MLP(
                layer_sizes=[input_size, action_size], activation_fn=mean_activation
            )
        else:
            raise NotImplementedError(
                f"{mean_network_type} hasn't been implemented for the mean network yet :("
            )

        if log_std_network_type == NetworkType.MLP:
            input_size = get_mlp_size(input_shape)
            self.log_std_network = MLP(
                layer_sizes=[input_size, action_size], activation_fn=log_std_activation
            )
        elif log_std_network_type == NetworkType.PARAMETER:
            self.log_std_network = T.nn.Parameter(
                T.ones(action_size) * log_std_init, requires_grad=True
            )

    def get_action_distribution(self, input: T.Tensor) -> T.distributions.Distribution:
        mean = self.mean_network(input)
        if isinstance(self.log_std_network, T.nn.Parameter):
            log_std = self.log_std_network
        else:
            log_std = self.log_std_network(input)
        clipped_log_std = T.clamp(log_std, self.min_log_std, self.max_log_std)
        return T.distributions.Normal(mean, clipped_log_std.exp())

    def forward(self, input: T.Tensor) -> T.Tensor:
        distribution = self.get_action_distribution(input)
        return distribution.sample()
