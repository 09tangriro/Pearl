"""Methods for estimating statistics"""

import numpy as np
import torch as T

from anvilrl.common.enumerations import TrajectoryType
from anvilrl.common.type_aliases import Tensor
from anvilrl.common.utils import numpy_to_torch, torch_to_numpy


def sample_forward_kl_divergence(
    target_dist_prob: Tensor, sample_dist_prob: Tensor, dtype: str = "torch"
) -> T.Tensor:
    """
    Sample forward KL Divergence element-wise
    https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd

    :param target_dist_prob: probabilities from target distribution, p(x)
    :param sample_dist_prob: probabilites from sample distribution, q(x)
    :return: element-wise kl approximation
    """
    ratio = target_dist_prob / sample_dist_prob
    if TrajectoryType(dtype.lower()) == TrajectoryType.TORCH:
        ratio = numpy_to_torch(ratio)
        return ratio * T.log(ratio) - (ratio - 1)
    elif TrajectoryType(dtype.lower()) == TrajectoryType.NUMPY:
        ratio = torch_to_numpy(ratio)
        return ratio * np.log(ratio) - (ratio - 1)
    else:
        raise ValueError(
            f"`dtype` flag should be 'torch' or 'numpy', but received {dtype}"
        )


def sample_reverse_kl_divergence(
    target_dist_prob: Tensor, sample_dist_prob: Tensor, dtype: str = "torch"
) -> T.Tensor:
    """
    Sample reverse KL Divergence element-wise
    https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd

    :param target_dist_prob: probabilities from target distribution, p(x)
    :param sample_dist_prob: probabilites from sample distribution, q(x)
    :return: element-wise kl approximation
    """
    ratio = target_dist_prob / sample_dist_prob
    if TrajectoryType(dtype.lower()) == TrajectoryType.TORCH:
        ratio = numpy_to_torch(ratio)
        return (ratio - 1) - T.log(ratio)
    elif TrajectoryType(dtype.lower()) == TrajectoryType.NUMPY:
        ratio = torch_to_numpy(ratio)
        return (ratio - 1) - np.log(ratio)
    else:
        raise ValueError(
            f"`dtype` flag should be 'torch' or 'numpy', but received {dtype}"
        )
