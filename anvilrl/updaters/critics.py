from abc import ABC, abstractmethod
from typing import Iterator, Type, Union

import numpy as np
import torch as T
from torch.nn.parameter import Parameter

from anvilrl.common.type_aliases import Tensor, UpdaterLog
from anvilrl.common.utils import numpy_to_torch
from anvilrl.models.actor_critics import ActorCritic, Critic
from anvilrl.signal_processing.return_estimators import soft_q_target


class BaseCriticUpdater(ABC):
    """
    The base class with pre-defined methods for derived classes

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
        self, model: Union[Critic, ActorCritic]
    ) -> Iterator[Parameter]:
        """Get the critic model parameters"""
        if isinstance(model, Critic):
            return model.parameters()
        else:
            critic_parameters = model.critic.parameters()
            if hasattr(model, "critic2"):
                critic_parameters += model.critic2.parameters()
            return critic_parameters

    def run_optimizer(
        self,
        optimizer: T.optim.Optimizer,
        loss: T.Tensor,
        critic_parameters: Iterator[Parameter],
    ) -> None:
        """Run an optimization step"""
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(critic_parameters, self.max_grad)
        optimizer.step()

    @abstractmethod
    def __call__(self) -> UpdaterLog:
        """Run an optimization step"""


class ValueRegression(BaseCriticUpdater):
    """
    Regression for a value function estimator

    :param loss_class: the distance loss class for regression, defaults to MSE
    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param loss_coeff: the coefficient for the value loss, defaults to 1
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        loss_class: T.nn.Module = T.nn.MSELoss(),
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 0.001,
        loss_coeff: float = 1,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.loss_class = loss_class
        self.loss_coeff = loss_coeff

    def __call__(
        self,
        model: Union[Critic, ActorCritic],
        observations: Tensor,
        returns: T.Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations: observation inputs
        :param returns: the target to regress to (e.g. TD Values, Monte-Carlo Values)
        """
        critic_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(critic_parameters, lr=self.lr)

        if isinstance(model, Critic):
            values = model(observations)
        else:
            values = model.forward_critic(observations)

        loss = self.loss_coeff * self.loss_class(values, returns)

        self.run_optimizer(optimizer, loss, critic_parameters)

        return UpdaterLog(loss=loss.detach().item())


class ContinuousQRegression(BaseCriticUpdater):
    """
    Regression for a continuous Q function estimator

    :param loss_class: the distance loss class for regression, defaults to MSE
    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param loss_coeff: the coefficient for the Q loss, defaults to 1
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        loss_class: T.nn.Module = T.nn.MSELoss(),
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 0.001,
        loss_coeff: float = 1,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.loss_class = loss_class
        self.loss_coeff = loss_coeff

    def __call__(
        self,
        model: Union[Critic, ActorCritic],
        observations: Tensor,
        actions: Tensor,
        returns: T.Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations: observation inputs
        :param actions: action inputs needed for continuous Q function modelling
        :param returns: the target to regress to (e.g. TD Values, Monte-Carlo Values)
        """
        critic_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(critic_parameters, lr=self.lr)

        if isinstance(model, Critic):
            q_values = model(observations, actions)
        else:
            q_values = model.forward_critic(observations, actions)

        loss = self.loss_coeff * self.loss_class(q_values, returns)

        self.run_optimizer(optimizer, loss, critic_parameters)

        return UpdaterLog(loss=loss.detach().item())


class DiscreteQRegression(BaseCriticUpdater):
    """
    Regression for a discrete Q function estimator

    :param loss_class: the distance loss class for regression, defaults to MSE
    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param lr: the learning rate for the optimizer algorithm
    :param loss_coeff: the coefficient for the Q loss, defaults to 1
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        loss_class: T.nn.Module = T.nn.MSELoss(),
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 0.001,
        loss_coeff: float = 1,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.loss_class = loss_class
        self.loss_coeff = loss_coeff

    def __call__(
        self,
        model: Union[Critic, ActorCritic],
        observations: Tensor,
        returns: T.Tensor,
        actions_index: Tensor,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations: observation inputs
        :param returns: the target to regress to (e.g. TD Values, Monte-Carlo Values)
        :param actions_index: discrete action values to use as indices, needed to filter
            the Q values for actions experienced.
        """
        critic_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(critic_parameters, lr=self.lr)

        if isinstance(model, Critic):
            q_values = model(observations)
        else:
            q_values = model.forward_critic(observations)
        actions_index = numpy_to_torch(actions_index)
        q_values = T.gather(q_values, dim=-1, index=actions_index.long())

        loss = self.loss_coeff * self.loss_class(q_values, returns)

        self.run_optimizer(optimizer, loss, critic_parameters)

        return UpdaterLog(loss=loss.detach().item())


class SoftQRegression(BaseCriticUpdater):
    def __init__(
        self,
        loss_class: T.nn.Module = T.nn.MSELoss(),
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 0.001,
        loss_coeff: float = 1,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class=optimizer_class, lr=lr, max_grad=max_grad)
        self.loss_class = loss_class
        self.loss_coeff = loss_coeff

    def __call__(
        self,
        model: ActorCritic,
        observations: T.Tensor,
        next_observations: T.Tensor,
        actions: T.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        alpha: float,
        gamma: float = 0.99,
    ) -> UpdaterLog:
        """
        Perform an optimization step

        :param model: the model on which the optimization should be run
        :param observations: observation inputs
        :param returns: the target to regress to (e.g. TD Values, Monte-Carlo Values)
        :param actions: optional action inputs, defaults to None, needed for continuous Q function modelling
        """
        critic_parameters = self._get_model_parameters(model)
        optimizer = self.optimizer_class(critic_parameters, lr=self.lr)

        next_action_distributions = model.get_action_distribution(next_observations)
        next_actions = next_action_distributions.rsample()
        log_probs = next_action_distributions.log_prob(next_actions)

        if hasattr(model, "target_critic"):
            if hasattr(model, "critic2"):
                q_values1 = model.target_critic(next_observations, next_actions)
                q_values2 = model.target_critic2(next_observations, next_actions)
                q_values = T.min(q_values1, q_values2)
            else:
                q_values = model.target_critic(next_observations, next_actions)
        else:
            q_values = model.critic(next_observations, next_actions)

        q_target = soft_q_target(
            rewards, dones, q_values.numpy(), log_probs.numpy(), alpha, gamma
        )

        loss = self.loss_coeff * self.loss_class(
            model.critic(observations, actions), q_target
        )
        if hasattr(model, "critic2"):
            loss += self.loss_coeff * self.loss_class(
                model.critic2(observations, actions), q_target
            )

        self.run_optimizer(optimizer, loss, critic_parameters)

        return UpdaterLog(loss=loss.detach().item())
