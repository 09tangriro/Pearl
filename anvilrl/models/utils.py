from typing import Optional, Tuple, Union

import torch as T

from anvilrl.common.type_aliases import Tensor
from anvilrl.common.utils import numpy_to_torch


def trainable_parameters(model: T.nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def prod(val):
    res = 1
    for ele in val:
        res *= ele
    return res


def get_mlp_size(data_shape: Union[int, Tuple[int]]) -> int:
    """
    Convert the shape of some data into an integer size that defines
    how many neurons to create in an MLP layer

    :param data_shape: the tuple shape or already converted integer size
    :return: the integer size
    """

    if isinstance(data_shape, tuple):
        data_shape = prod(list(data_shape))
    return data_shape


def concat_obs_actions(observations: Tensor, actions: Optional[Tensor]) -> T.Tensor:
    if actions is not None:
        observations, actions = numpy_to_torch(observations, actions)
        return T.cat([observations, actions], dim=-1)
    return numpy_to_torch(observations)
