from typing import Optional, Tuple, Union

import torch as T

from pearll import settings
from pearll.common.type_aliases import Tensor


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


def preprocess_inputs(observations: Tensor, actions: Optional[Tensor]) -> T.Tensor:
    input = T.as_tensor(observations)
    if input.dim() == 0:
        input = input.unsqueeze(0)
    if actions is not None:
        actions = T.as_tensor(actions)
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        input = T.cat([input, actions], dim=-1)
    return input.float().to(settings.DEVICE, non_blocking=True)
