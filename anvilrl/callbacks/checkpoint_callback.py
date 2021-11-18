import os

import torch as T

from anvilrl.callbacks.base_callback import BaseCallback
from anvilrl.common.logging_ import Logger
from anvilrl.models.actor_critics import ActorCritic


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        logger: Logger,
        save_freq: int,
        save_path: str,
        model: ActorCritic,
        name_prefix: str = "agent",
    ) -> None:
        super().__init__(logger, model)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        os.makedirs(save_path, exist_ok=True)

    def load(self, path: str):
        """Load the model"""
        path = path + ".pt"
        self.logger.info(f"Loading weights from {path}")
        try:
            self.model.load_state_dict(T.load(path))
        except FileNotFoundError:
            self.logger.info("File not found, assuming no model dict was to be loaded")

    def save(self, path: str):
        """Save the model"""
        path = path + ".pt"
        self.logger.info(f"Saving weights to {path}")
        T.save(self.model.state_dict(), path)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.step}_steps")
            self.save(path)
        return True
