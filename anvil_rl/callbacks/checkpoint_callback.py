import os

from anvil_rl.callbacks.base_callback import BaseCallback


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: bool = True,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.n_calls}_steps"
            )
            self.agent.save(path)
        return True
