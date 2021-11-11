from abc import ABC, abstractmethod

from anvilrl.common.logging_ import Logger
from anvilrl.models.actor_critics import ActorCritic


class BaseCallback(ABC):
    """
    Base class for callback.
    :param logger:
    """

    def __init__(self, logger: Logger, model: ActorCritic) -> None:
        self.n_calls = 0
        self.step = 0
        self.logger = logger
        self.model = model

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """

    def on_step(self, step: int) -> bool:
        """
        This method will be called by the model after each call to ``step_env()``.
        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.
        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.step = step

        return self._on_step()
