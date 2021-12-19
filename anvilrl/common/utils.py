import random
from dataclasses import asdict
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch as T
from gym import Env, spaces


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


def numpy_to_torch(*data, **kwargs) -> Union[Tuple[T.Tensor], T.Tensor]:
    """Convert any numpy arrays into torch tensors"""
    device = get_device(kwargs.pop("device", "auto"))
    result = [None] * len(data)
    for i, el in enumerate(data):
        if isinstance(el, np.ndarray):
            result[i] = T.Tensor(el).to(device)
        elif isinstance(el, T.Tensor):
            result[i] = el.to(device)
        else:
            raise RuntimeError(
                f"{type(el)} is not a recognized data format, it should be a numpy array or torch tensor"
            )
    if len(data) == 1:
        return result[0]
    else:
        return tuple(result)


def torch_to_numpy(*data) -> Union[Tuple[np.ndarray], np.ndarray]:
    """Convert any torch tensors into numpy arrays"""
    result = [None] * len(data)
    for i, el in enumerate(data):
        if isinstance(el, T.Tensor):
            result[i] = el.detach().numpy()
        elif isinstance(el, np.ndarray):
            result[i] = el
        else:
            raise RuntimeError(
                f"{type(el)} is not a recognized data format, it should be a numpy array or torch tensor"
            )

    if len(data) == 1:
        return result[0]
    else:
        return tuple(result)


def get_space_shape(
    space: spaces.Space,
) -> Tuple[int, ...]:
    """
    Get the shape of a space (useful for the buffers).
    :param space:
    :return:
    """
    if isinstance(space, spaces.Tuple):
        return (len(space),) + get_space_shape(space.spaces[0])
    if isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        # an int
        return (1,)
    elif isinstance(space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(space.nvec)),)
    elif isinstance(space, spaces.MultiBinary):
        # Number of binary features
        return (int(space.n),)
    elif isinstance(space, spaces.Dict):
        return get_space_shape(space["observation"])
    else:
        raise NotImplementedError(f"{space} observation space is not supported")


def get_space_range(space: spaces.Space) -> Tuple[Any, Any]:
    """
    Get the range of a space (useful for the buffers).
    :param space:
    :return:
    """
    if isinstance(space, spaces.Tuple):
        return get_space_range(space.spaces[0])
    if isinstance(space, spaces.Box):
        return space.low, space.high
    elif isinstance(space, spaces.Discrete):
        return 0, space.n - 1
    elif isinstance(space, spaces.MultiDiscrete):
        return np.zeros(space.nvec.shape), space.nvec - 1
    elif isinstance(space, spaces.MultiBinary):
        return 0, 1
    elif isinstance(space, spaces.Dict):
        return get_space_range(space["observation"])
    else:
        raise NotImplementedError(f"{space} observation space is not supported")


def extend_shape(original_shape: Tuple, new_size: int, axis: int = 0) -> Tuple:
    """Extend a dimension of a shape tuple"""

    shape = list(original_shape)
    shape[axis] = new_size
    return tuple(shape)


def filter_dataclass_by_none(class_object: Any) -> Dict[str, Any]:
    dict_class = asdict(class_object)
    return {k: v for k, v in dict_class.items() if v is not None}


def set_seed(seed: int, env: Env) -> None:
    """
    Set the seed for all the random generators.
    :param seed: The seed to set
    :param env: The environment to set the seed for
    """
    random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
