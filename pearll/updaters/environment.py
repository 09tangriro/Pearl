from abc import ABC, abstractmethod
from typing import Iterator, Type

import torch as T
from torch.nn import functional as F

from pearll.common.type_aliases import Tensor, UpdaterLog
from pearll.models.actor_critics import Model


class BaseDeepUpdater(ABC):
    """
    Base class for updating a deep environment model.
    """

    def __init__(
        self,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        max_grad: float = 0,
    ) -> None:
        self.optimizer_class = optimizer_class
        self.max_grad = max_grad

    def run_optimizer(
        self,
        optimizer: T.optim.Optimizer,
        loss: T.Tensor,
        model_parameters: Iterator[T.nn.Parameter],
    ) -> None:
        """Run an optimization step"""
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            T.nn.utils.clip_grad_norm_(model_parameters, self.max_grad)
        optimizer.step()

    @abstractmethod
    def __call__(
        self,
        model: Model,
        predictions: T.Tensor,
        targets: T.Tensor,
        learning_rate: float = 0.001,
    ) -> UpdaterLog:
        """
        Run an optimization step

        :param model: The model to update (observation, reward or done models)
        :param predictions: The predictions to update
        :param targets: The targets to regress against
        :param learning_rate: The learning rate to use
        """


class DeepRegression(BaseDeepUpdater):
    """
    Update a deep model using a deep regression algorithm.

    :param loss_class: The loss class to use e.g. MSE
    :param optimizer_class: the type of optimizer to use, defaults to Adam
    :param max_grad: maximum gradient clip value, defaults to no clipping with a value of 0
    """

    def __init__(
        self,
        loss_class: T.nn.Module = T.nn.MSELoss(),
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        max_grad: float = 0,
    ) -> None:
        super().__init__(optimizer_class, max_grad)
        self.loss_class = loss_class

    def __call__(
        self,
        model: Model,
        observations: Tensor,
        actions: Tensor,
        targets: T.Tensor,
        learning_rate: float = 0.001,
        mode: str = "auto",
    ) -> UpdaterLog:
        """
        Run an optimization step

        :param model: The model to update (observation, reward or done models)
        :param predictions: The predictions to update
        :param targets: The targets to regress against
        :param learning_rate: The learning rate to use
        :param mode: The mode to use, defaults to auto. If set to anything else, no processing
            will be done on the targets and predictions for different loss functions.
        """
        params = model.parameters()
        predictions = model(observations, actions)

        if mode == "auto":
            # Data processing for categorical losses; assumes categorical target.
            if isinstance(self.loss_class, (T.nn.CrossEntropyLoss, T.nn.BCELoss)):
                torso_output = model.torso(model.encoder(observations, actions))
                predictions = super(type(model.head), model.head).forward(torso_output)
                targets = (
                    F.one_hot(targets.long(), predictions.shape[-1]).squeeze().float()
                )

        optimizer = self.optimizer_class(params, lr=learning_rate)
        loss = self.loss_class(predictions, targets)
        self.run_optimizer(optimizer, loss, params)

        return UpdaterLog(loss=loss.detach().item())
