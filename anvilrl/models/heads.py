from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, Union

import torch as T

from anvilrl.common.enumerations import NetworkType
from anvilrl.models.torsos import MLP
from anvilrl.models.utils import get_mlp_size

################################### BASE CLASSES ###################################


class BaseCriticHead(T.nn.Module):
    """The base class for the critic head network"""

    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, input: T.Tensor) -> T.Tensor:
        return self.model(input)


class BaseActorHead(T.nn.Module, ABC):
    """The base class for the actor head network"""

    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def get_action_distribution(
        self, input: T.Tensor
    ) -> Optional[T.distributions.Distribution]:
        """
        Get the action distribution given a latent input from the torso

        :param input: torso network output
        :return: the distribution of the actor, returns None if deterministic
        """

    def forward(self, input: T.Tensor) -> T.Tensor:
        distribution = self.get_action_distribution(input)
        if distribution is None:
            return self.model(input)
        return distribution.sample()


################################### CRITIC HEADS ###################################


class ValueHead(BaseCriticHead):
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
        activation_fn: Optional[Type[T.nn.Module]] = None,
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


class DiscreteQHead(BaseCriticHead):
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
        activation_fn: Optional[Type[T.nn.Module]] = None,
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


################################### ACTOR HEADS ###################################


class DeterministicHead(BaseActorHead):
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
        activation_fn: Optional[Type[T.nn.Module]] = None,
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

    def get_action_distribution(
        self, input: T.Tensor
    ) -> Optional[T.distributions.Distribution]:
        return None


class CategoricalHead(BaseActorHead):
    """
    Use this head for a categorical actor.

    :param input_shape: the input shape to the head network, can be the tuple shape or simplified integer input size
    :param action_size: the dimension of the action vector
    :param network_type: the type of network used
    :param activation_fn: the activation function after each layer
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        action_size: int,
        network_type: str = "mlp",
        activation_fn: Optional[Type[T.nn.Module]] = None,
    ):
        super().__init__()
        network_type = NetworkType(network_type.lower())
        if network_type == NetworkType.MLP:
            input_size = get_mlp_size(input_shape)
            self.model = MLP(
                layer_sizes=[input_size, action_size], activation_fn=activation_fn
            )
        else:
            raise NotImplementedError(f"{network_type} hasn't been implemented yet :(")

    def get_action_distribution(
        self, input: T.Tensor
    ) -> Optional[T.distributions.Distribution]:
        return T.distributions.Categorical(logits=self.model(input))


class DiagGaussianHead(BaseActorHead):
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
        else:
            raise NotImplementedError(
                f"{log_std_network_type} hasn't been implemented for the log std network yet :("
            )

    def get_action_distribution(
        self, input: T.Tensor
    ) -> Optional[T.distributions.Distribution]:
        mean = self.mean_network(input)
        if isinstance(self.log_std_network, T.nn.Parameter):
            log_std = self.log_std_network
        else:
            log_std = self.log_std_network(input)
        if self.min_log_std or self.max_log_std:
            log_std = T.clamp(log_std, self.min_log_std, self.max_log_std)
        return T.distributions.Normal(mean, log_std.exp())
