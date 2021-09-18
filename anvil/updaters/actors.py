from typing import Optional, Type, Union

import torch as T

from anvil.common.type_aliases import ActorUpdaterLog
from anvil.models.actor_critics import Actor, ActorCritic
from anvil.updaters.utils import sample_reverse_kl_divergence


class PolicyGradient(object):
    """
    Vanilla policy gradient with entropy regulation: https://spinningup.openai.com/en/latest/algorithms/vpg.html
    loss = -E[A(state,action) * log(policy(action|state)) + entropy_coeff * entropy(policy)]

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param entropy_coeff: entropy regulation coefficient
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        entropy_coeff: float = 0.01,
        max_grad: float = 0,
    ) -> None:
        self.entropy_coeff = entropy_coeff
        self.max_grad = max_grad
        self.optimizer_class = optimizer_class
        self.lr = lr

    def __call__(
        self,
        model: Union[ActorCritic, Actor],
        observations: T.Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        log_probs: Optional[T.Tensor] = None,
    ) -> ActorUpdaterLog:
        """
        Perform and optimization step

        :param model: the model on which the optimization should be run
        :param observations: observations
        :param actions: actions
        :param advantages: advantage function
        :param log_probs: log probability of observing actions given the observations
        """
        optimizer = self.optimizer_class(model.parameters(), lr=self.lr)
        distributions = model.get_action_distribution(observations)
        new_log_probs = distributions.log_prob(actions).sum(dim=-1)
        entropy = distributions.entropy().mean()

        batch_loss = -(advantages * new_log_probs).mean()
        entropy_loss = -self.entropy_coeff * entropy

        loss = batch_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad)
        optimizer.step()

        loss = loss.detach()
        entropy = entropy.detach()
        if log_probs is not None:
            kl = sample_reverse_kl_divergence(
                log_probs.exp().detach(), new_log_probs.exp().detach()
            )
        else:
            kl = None

        return ActorUpdaterLog(loss=loss, kl=kl, entropy=entropy)


class ProximalPolicyOptimization(object):
    def __init__(
        self,
        model: Union[ActorCritic, Actor],
        ratio_clip: float = 0.2,
        max_kl: float = 0.015,
        entropy_coeff: float = 0.01,
        max_grad: float = 0,
    ) -> None:
        self.model = model
        self.ratio_clip = ratio_clip
        self.max_kl = max_kl
        self.entropy_coeff = entropy_coeff
        self.max_grad = max_grad

    def __call__(
        self,
        observations: T.Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        log_probs: T.Tensor,
    ):
        pass
