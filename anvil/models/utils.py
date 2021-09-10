from enum import Enum
from typing import Union

import torch as T


class NetworkType(Enum):
    MLP = "mlp"
    IDENTITY = "identity"


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
