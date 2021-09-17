import torch as T


def sample_forward_kl_divergence(
    target_dist_prob: T.Tensor, sample_dist_prob: T.Tensor
) -> T.Tensor:
    ratio = target_dist_prob / sample_dist_prob
    return ratio * T.log(ratio) - (ratio - 1)


def sample_reverse_kl_divergence(
    target_dist_prob: T.Tensor, sample_dist_prob: T.Tensor
) -> T.Tensor:
    ratio = target_dist_prob / sample_dist_prob
    return (ratio - 1) - T.log(ratio)
