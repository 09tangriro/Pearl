from enum import Enum
from typing import Tuple, Union

import torch as T


class NetworkType(Enum):
    MLP = "mlp"
    PARAMETER = "parameter"


def get_device(device: Union[T.device, str]) -> T.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = T.device(device)

    # Cuda not available
    if device.type == T.device("cuda").type and not T.cuda.is_available():
        return T.device("cpu")

    return device


def get_mlp_size(data_shape: Union[int, Tuple[int]]) -> int:
    """
    Convert the shape of some data into an integer size that defines
    how many neurons to create in an MLP layer

    :param data_shape: the tuple shape or already converted integer size
    :return: the integer size
    """

    if isinstance(data_shape, tuple):
        data_shape = data_shape[0]
    return data_shape
