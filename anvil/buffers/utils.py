from typing import Dict, Tuple, Union

from gym import spaces


def get_space_shape(
    space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of a space (useful for the buffers).
    :param space:
    :return:
    """
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
        return {
            key: get_space_shape(subspace) for (key, subspace) in space.spaces.items()
        }

    else:
        raise NotImplementedError(f"{space} observation space is not supported")
