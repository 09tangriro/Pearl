from abc import ABC, abstractmethod
from typing import Iterator, Type, Union

import torch as T
from torch.distributions import kl_divergence
from torch.nn.parameter import Parameter

from anvilrl.common.type_aliases import Tensor, UpdaterLog
from anvilrl.models.actor_critics import Actor, ActorCritic


class BaseActorUpdater(ABC):
    """
    The base actor updater class with pre-defined methods for derived classes

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

    def _get_model_parameters(
        self, model: Union[Actor, ActorCritic]
    ) -> Iterator[Parameter]:
        """Get the actor model parameters"""
        if isinstance(model, Actor):
            return model.parameters()
        else:
            return model.actor.parameters()

    def run_optimizer(
        self,
        optimizer: T.optim.Optimizer,
        loss: T.Tensor,
        actor_parameters: Iterator[Parameter],
    ) -> None:
        """Run an optimization step"""
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(actor_parameters, self.max_grad)
        optimizer.step()

    @abstractmethod
    def __call__(self) -> UpdaterLog:
        """Run an optimization step"""


class PolicyGradient(BaseActorUpdater):
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
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.entropy_coeff = entropy_coeff

    def __call__(
        self,
        model: Union[ActorCritic, Actor],
        observations: Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations:
        :param actions:
        :param advantages:
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=self.lr)
        old_distributions = model.get_action_distribution(observations)
        log_probs = old_distributions.log_prob(actions).sum(dim=-1)
        entropy = old_distributions.entropy().mean()

        batch_loss = -(advantages * log_probs).mean()
        entropy_loss = -self.entropy_coeff * entropy

        loss = batch_loss + entropy_loss

        self.run_optimizer(optimizer, loss, actor_parameters)

        new_distributions = model.get_action_distribution(observations)
        loss = loss.detach()
        entropy = entropy.detach()
        kl = kl_divergence(new_distributions, old_distributions).mean()

        return UpdaterLog(
            loss=loss.item(), divergence=kl.item(), entropy=entropy.item()
        )


class ProximalPolicyClip(BaseActorUpdater):
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
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.ratio_clip = ratio_clip
        self.max_kl = max_kl
        self.entropy_coeff = entropy_coeff

    def __call__(
        self,
        model: Union[ActorCritic, Actor],
        observations: Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        old_log_probs: T.Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations:
        :param actions:
        :param advantages:
        :param old_log_probs:
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=self.lr)
        old_distributions = model.get_action_distribution(observations)
        log_probs = old_distributions.log_prob(actions).sum(dim=-1)
        entropy = old_distributions.entropy().mean()

        ratios = (log_probs - old_log_probs).exp()

        raw_loss = ratios * advantages
        clipped_loss = (
            T.clamp(ratios, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantages
        )

        batch_loss = -(T.min(raw_loss, clipped_loss)).mean()
        entropy_loss = -self.entropy_coeff * entropy

        loss = batch_loss + entropy_loss

        self.run_optimizer(optimizer, loss, actor_parameters)

        new_distributions = model.get_action_distribution(observations)
        loss = loss.detach()
        entropy = entropy.detach()
        kl = kl_divergence(new_distributions, old_distributions).mean()

        return UpdaterLog(
            loss=loss.item(), divergence=kl.item(), entropy=entropy.item()
        )


class DeterministicPolicyGradient(BaseActorUpdater):
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
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)

    def __call__(
        self,
        model: ActorCritic,
        observations: Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: an actor critic model
        :param observations:
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=self.lr)

        actions = model(observations)
        values = model.critic(observations, actions)

        loss = -values.mean()

        self.run_optimizer(optimizer, loss, actor_parameters)

        return UpdaterLog(loss=loss.detach().item())


class SoftPolicyGradient(BaseActorUpdater):
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
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.entropy_coeff = entropy_coeff
        self.squashed_output = squashed_output

    def __call__(
        self,
        model: ActorCritic,
        observations: Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: an actor critic model with either 1 or 2 critic networks
        :param observations:
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=self.lr)

        distributions = model.get_action_distribution(observations)
        # use the reparametrization trick for backpropagation
        # https://gregorygundersen.com/blog/2018/04/29/reparameterization/
        actions = distributions.rsample()
        if self.squashed_output:
            actions = T.tanh(actions)
        log_probs = distributions.log_prob(actions)
        entropy = distributions.entropy().mean()

        with T.no_grad():
            if hasattr(model, "critic2"):
                values1, values2 = model.forward_critic(observations, actions)
                values = T.min(values1, values2)
            else:
                values = model.forward_critic(observations, actions)

        loss = (self.entropy_coeff * log_probs - values).mean()

        self.run_optimizer(optimizer, loss, actor_parameters)

        return UpdaterLog(loss=loss.detach().item(), entropy=entropy.detach().item())
