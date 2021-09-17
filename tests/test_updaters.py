import pytest
import torch as T

from anvil.updaters.utils import (
    sample_forward_kl_divergence,
    sample_reverse_kl_divergence,
)


@pytest.mark.parametrize(
    "kl_divergence", [sample_forward_kl_divergence, sample_reverse_kl_divergence]
)
def test_kl_divergence(kl_divergence):
    # KL Div = E(unbiased approx KL Div)
    T.manual_seed(8)
    num_samples = int(2e6)
    dist1 = T.distributions.Normal(0, 1)
    dist2 = T.distributions.Normal(1, 2)

    samples = dist2.sample((num_samples,))

    log_probs1 = dist1.log_prob(samples)
    log_probs2 = dist2.log_prob(samples)

    if kl_divergence == sample_forward_kl_divergence:
        full_kl_div = T.distributions.kl_divergence(dist1, dist2)
    else:
        full_kl_div = T.distributions.kl_divergence(dist2, dist1)

    approx_kl_div = kl_divergence(log_probs1.exp(), log_probs2.exp()).mean()

    assert T.allclose(full_kl_div, approx_kl_div, rtol=5e-3)
