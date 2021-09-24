from typing import Optional, Type, Union

import torch as T
from numpy import log

from anvil.common.type_aliases import ActorUpdaterLog
from anvil.models.actor_critics import Actor, ActorCritic, Critic
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
        old_log_probs: Optional[T.Tensor] = None,
    ) -> ActorUpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations:
        :param actions:
        :param advantages:
        :param old_log_probs:
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
        if old_log_probs is not None:
            kl = sample_reverse_kl_divergence(
                old_log_probs.exp().detach(), new_log_probs.exp().detach()
            )
        else:
            kl = None

        return ActorUpdaterLog(loss=loss, kl=kl, entropy=entropy)


class ProximalPolicyClip(object):
    """
    PPO-Clip algorithm with entropy regularization: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    loss = E[min(r)]

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param ratio_clip: the clipping factor for the clipped loss
    :param max_kl: the maximum kl divergence between old and new policies in an update step
    :param entropy_coeff: entropy regulation coefficient
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        ratio_clip: float = 0.2,
        max_kl: float = 0.015,
        entropy_coeff: float = 0.01,
        max_grad: float = 0,
    ) -> None:
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.ratio_clip = ratio_clip
        self.max_kl = max_kl
        self.entropy_coeff = entropy_coeff
        self.max_grad = max_grad

    def __call__(
        self,
        model: Union[ActorCritic, Actor],
        observations: T.Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        old_log_probs: T.Tensor,
    ):
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations:
        :param actions:
        :param advantages:
        :param old_log_probs:
        """
        optimizer = self.optimizer_class(model.parameters(), lr=self.lr)
        distributions = model.get_action_distribution(observations)
        new_log_probs = distributions.log_prob(actions).sum(dim=-1)
        entropy = distributions.entropy().mean()

        ratios = (new_log_probs - old_log_probs).exp()

        raw_loss = ratios * advantages
        clipped_loss = (
            T.clamp(ratios, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantages
        )

        batch_loss = -(T.min(raw_loss, clipped_loss)).mean()
        entropy_loss = -self.entropy_coeff * entropy

        loss = batch_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad)
        optimizer.step()

        loss = loss.detach()
        entropy = entropy.detach()
        if old_log_probs is not None:
            kl = sample_reverse_kl_divergence(
                old_log_probs.exp().detach(), new_log_probs.exp().detach()
            )
        else:
            kl = None

        return ActorUpdaterLog(loss=loss, kl=kl, entropy=entropy)


class DeterministicPolicyGradient(object):
    """
    Deterministic policy gradient used in DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    loss = -E[critic(state, actor(state))]

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        max_grad: float = 0,
    ) -> None:
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.max_grad = max_grad

    def __call__(
        self,
        actor: Actor,
        critic: Critic,
        observations: T.Tensor,
    ) -> ActorUpdaterLog:
        """
        Perform an optimization step

        :param actor: the actor model or sub-model
        :param critic: the critic model or sub-model
        :param observations:
        """
        optimizer = self.optimizer_class(actor.parameters(), lr=self.lr)
        # make sure critic isn't updated!
        for var in critic.parameters():
            var.requires_grad = False

        actions = actor(observations)
        values = critic(observations, actions)

        loss = -values.mean()

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad)
        optimizer.step()

        # reset critic parameters
        for var in critic.parameters():
            var.requires_grad = True

        return ActorUpdaterLog(loss=loss)


class SoftPolicyGradient(object):
    """
    Policy gradient update used in SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param entropy_coeff: entropy weighting coefficient
    :param squashed_output: whether to squash the actions using tanh, defaults to True
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        entropy_coeff: float = 0.2,
        squashed_output: bool = True,
        max_grad: float = 0,
    ) -> None:
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.entropy_coeff = entropy_coeff
        self.max_grad = max_grad
        self.squashed_output = squashed_output

    def __call__(
        self,
        actor: Actor,
        critic1: Critic,
        critic2: Optional[Critic],
        observations: T.Tensor,
    ) -> ActorUpdaterLog:
        """
        Perform an optimization step

        :param actor: the actor model or sub-model
        :param critic1: the first critic model or sub-model
        :param critic2: optional second critic model or sub-model
        :param observations:
        """
        optimizer = self.optimizer_class(actor.parameters(), lr=self.lr)
        # make sure critic isn't updated!
        critic_variables = critic1.parameters() + critic2.parameters()
        for var in critic_variables:
            var.requires_grad = False

        distributions = actor.get_action_distribution(observations)
        # use the reparametrization trick for backpropagation
        # https://gregorygundersen.com/blog/2018/04/29/reparameterization/
        actions = distributions.rsample()
        if self.squashed_output:
            actions = T.tanh(actions)
        log_probs = distributions.log_prob(actions)
        entropy = distributions.entropy().mean()

        values1 = critic1(observations, actions)
        if critic2 is not None:
            values2 = critic2(observations, actions)
            values = T.min(values1, values2)
        else:
            values = values1

        loss = (self.entropy_coeff * log_probs - values).mean()

        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad)
        optimizer.step()

        # reset critic parameters
        for var in critic_variables:
            var.requires_grad = True

        return ActorUpdaterLog(loss=loss, entropy=entropy)
