import os
from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import torch as T
from gym import Env

from anvil.buffers.base_buffer import BaseBuffer
from anvil.models.actor_critics import ActorCritic
from anvil.signal_processing.explorers import BaseExplorer
from anvil.updaters.actors import BaseActorUpdater
from anvil.updaters.critics import BaseCriticUpdater

logger = Logger(__name__)


class BaseAgent(ABC):
    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        actor_updater: BaseActorUpdater,
        critic_updater: BaseCriticUpdater,
        buffer: BaseBuffer,
        action_explorer: Optional[BaseExplorer] = None,
    ) -> None:
        self.env = env
        self.model = model
        self.actor_updater = actor_updater
        self.critic_updater = critic_updater
        self.buffer = buffer
        self.action_explorer = action_explorer

    def save(self, path: str):
        """Save the model"""
        path = path + ".pt"
        logger.info(f"Saving weights to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load the model"""
        path = path + ".pt"
        logger.info(f"Loading weights from {path}")
        self.model.load_state_dict(T.load(path))

    @abstractmethod
    def fit(self, num_steps: int, eval_freq: int = -1):
        """Train the agent in the environment"""
