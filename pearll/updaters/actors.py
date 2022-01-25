from abc import ABC, abstractmethod
from typing import Iterator, Type, Union

import torch as T
from torch.distributions import kl_divergence
from torch.nn.parameter import Parameter

from pearll.common.type_aliases import Tensor, UpdaterLog
from pearll.models.actor_critics import Actor, ActorCritic


class BaseActorUpdater(ABC):
    """
    The base actor updater class with pre-defined methods for derived classes

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        max_grad: float = 0,
    ) -> None:
        self.optimizer_class = optimizer_class
        self.max_grad = max_grad

    def _get_model_parameters(
        self, model: Union[Actor, ActorCritic]
    ) -> Iterator[Parameter]:
        """Get the actor model parameters"""
        if isinstance(model, Actor):
            return model.parameters()
        else:
            params = []
            for actor in model.actors:
                params.extend(actor.model.parameters())
            return params

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
    def __call__(self, model: Union[Actor, ActorCritic]) -> UpdaterLog:
        """Run an optimization step"""


class PolicyGradient(BaseActorUpdater):
    """
    Vanilla policy gradient with entropy regulation: https://spinningup.openai.com/en/latest/algorithms/vpg.html
    loss = -E[A(state,action) * log(policy(action|state)) + entropy_coeff * entropy(policy)]

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, max_grad=max_grad)

    def __call__(
        self,
        model: Union[ActorCritic, Actor],
        observations: Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        learning_rate: float = 1e-3,
        entropy_coeff: float = 0.01,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations:
        :param actions:
        :param advantages:
        :param learning_rate: the learning rate for the optimizer algorithm
        :param entropy_coeff: entropy regulation coefficient
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=learning_rate)
        old_distributions = model.action_distribution(observations)
        log_probs = old_distributions.log_prob(actions).sum(dim=-1)
        entropy = old_distributions.entropy().mean()

        batch_loss = -(advantages * log_probs).mean()
        entropy_loss = -entropy_coeff * entropy

        loss = batch_loss + entropy_loss

        self.run_optimizer(optimizer, loss, actor_parameters)

        new_distributions = model.action_distribution(observations)
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
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, max_grad=max_grad)

    def __call__(
        self,
        model: Union[ActorCritic, Actor],
        observations: Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        old_log_probs: T.Tensor,
        learning_rate: float = 1e-3,
        ratio_clip: float = 0.2,
        entropy_coeff: float = 0.01,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations:
        :param actions:
        :param advantages:
        :param old_log_probs:
        :param learning_rate: the learning rate for the optimizer algorithm
        :param ratio_clip: the clipping factor for the clipped loss
        :param entropy_coeff: entropy regulation coefficient
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=learning_rate)
        old_distributions = model.action_distribution(observations)
        log_probs = old_distributions.log_prob(actions).sum(dim=-1)
        entropy = old_distributions.entropy().mean()

        ratios = (log_probs - old_log_probs).exp()

        raw_loss = ratios * advantages
        clipped_loss = T.clamp(ratios, 1 - ratio_clip, 1 + ratio_clip) * advantages

        batch_loss = -(T.min(raw_loss, clipped_loss)).mean()
        entropy_loss = -entropy_coeff * entropy

        loss = batch_loss + entropy_loss

        self.run_optimizer(optimizer, loss, actor_parameters)

        new_distributions = model.action_distribution(observations)
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
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, max_grad=max_grad)

    def __call__(
        self,
        model: ActorCritic,
        observations: Tensor,
        learning_rate: float = 1e-3,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: an actor critic model
        :param observations:
        :param learning_rate: the learning rate for the optimizer algorithm
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=learning_rate)

        actions = model(observations)
        values = model.forward_critics(observations, actions)

        loss = -values.mean()

        self.run_optimizer(optimizer, loss, actor_parameters)

        return UpdaterLog(loss=loss.detach().item())


class SoftPolicyGradient(BaseActorUpdater):
    """
    Policy gradient update used in SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param squashed_output: whether to squash the actions using tanh, defaults to True
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        squashed_output: bool = True,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, max_grad=max_grad)
        self.squashed_output = squashed_output

    def __call__(
        self,
        model: ActorCritic,
        observations: Tensor,
        learning_rate: float = 1e-3,
        entropy_coeff: float = 0.01,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: an actor critic model with either 1 or 2 critic networks
        :param observations:
        :param learning_rate: the learning rate for the optimizer algorithm
        :param entropy_coeff: entropy weighting coefficient
        """
        actor_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(actor_parameters, lr=learning_rate)

        distributions = model.action_distribution(observations)
        # use the reparametrization trick for backpropagation
        # https://gregorygundersen.com/blog/2018/04/29/reparameterization/
        actions = distributions.rsample()
        if self.squashed_output:
            actions = T.tanh(actions)
        log_probs = distributions.log_prob(actions)
        entropy = distributions.entropy().mean()

        with T.no_grad():
            if hasattr(model, "critic2"):
                values1, values2 = model.forward_critics(observations, actions)
                values = T.min(values1, values2)
            else:
                values = model.forward_critics(observations, actions)

        loss = (entropy_coeff * log_probs - values).mean()

        self.run_optimizer(optimizer, loss, actor_parameters)

        return UpdaterLog(loss=loss.detach().item(), entropy=entropy.detach().item())
