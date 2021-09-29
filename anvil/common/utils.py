from typing import Tuple, Union

import numpy as np
import torch as T


def get_device(device: Union[T.device, str]) -> T.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    if isinstance(device, T.device):
        return device

    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = T.device(device)

    # Cuda not available
    if device.type == T.device("cuda").type and not T.cuda.is_available():
        return T.device("cpu")

    return device


def numpy_to_torch(*data) -> Tuple[T.Tensor]:
    """Convert any numpy arrays into torch tensors"""
    result = [None] * len(data)
    for i, el in enumerate(data):
        if isinstance(el, np.ndarray):
            result[i] = T.Tensor(el)
        elif isinstance(el, T.Tensor):
            result[i] = el
        else:
            raise RuntimeError(
                f"{type(el)} is not a recognized data format, it should be a numpy array or torch tensor"
            )

    return tuple(result)


def torch_to_numpy(*data) -> Tuple[np.ndarray]:
    """Convert any torch tensors into numpy arrays"""
    result = [None] * len(data)
    for i, el in enumerate(data):
        if isinstance(el, T.Tensor):
            result[i] = el.numpy()
        elif isinstance(el, np.ndarray):
            result[i] = el
        else:
            raise RuntimeError(
                f"{type(el)} is not a recognized data format, it should be a numpy array or torch tensor"
            )

    return tuple(result)
