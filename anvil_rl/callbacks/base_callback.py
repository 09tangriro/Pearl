from abc import ABC, abstractmethod
from logging import INFO, Logger
from typing import Any, Dict, Optional

from gym import Env

from anvil_rl.agents import base_agent
from anvil_rl.models.actor_critics import ActorCritic


class BaseCallback(ABC):
    """
    Base class for callback.
    :param verbose:
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.n_calls = 0

    # Type hint as string to avoid circular import
    def init(self, agent: "base_agent.BaseAgent") -> None:
        self.logger = agent.logger
        self.agent = agent

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``step_env()``.
        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.
        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1

        return self._on_step()
