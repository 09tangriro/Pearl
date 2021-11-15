import numpy as np
import torch as T

from anvilrl.common.type_aliases import Tensor


def sample_forward_kl_divergence(
    target_dist_prob: Tensor, sample_dist_prob: Tensor
) -> T.Tensor:
    """
    Sample forward KL Divergence element-wise
    https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd

    :param target_dist_prob: probabilities from target distribution, p(x)
    :param sample_dist_prob: probabilites from sample distribution, q(x)
    :return: element-wise kl approximation
    """
    ratio = target_dist_prob / sample_dist_prob
    return ratio * T.log(ratio) - (ratio - 1)


def sample_reverse_kl_divergence(
    target_dist_prob: Tensor, sample_dist_prob: Tensor
) -> T.Tensor:
    """
    Sample reverse KL Divergence element-wise
    https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd

    :param target_dist_prob: probabilities from target distribution, p(x)
    :param sample_dist_prob: probabilites from sample distribution, q(x)
    :return: element-wise kl approximation
    """
    ratio = target_dist_prob / sample_dist_prob
    return (ratio - 1) - T.log(ratio)
