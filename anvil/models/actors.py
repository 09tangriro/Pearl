from typing import List, Type, Union

import torch as T

from anvil.models.heads import BaseActorHead
from anvil.models.utils import get_device


class Actor(T.nn.Module):
    """
    The actor network which determines what actions to take in the environment.

    :param encoder: the encoder network
    :param torso: the torso network
    :param heads: a list of network heads
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        head: BaseActorHead,
        optimizer_class: Type[T.optim.Optimizer] = T.optim.Adam,
        lr: float = 1e-3,
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.head = head.to(device)
        self.optimizer = optimizer_class(self.parameters(), lr=lr)

    def get_action_distribution(self, *inputs) -> T.distributions.Distribution:
        latent_out = self.torso(self.encoder(*inputs))
        return self.head.get_action_distribution(latent_out)

    def forward(self, *inputs) -> List[T.Tensor]:
        out = self.encoder(*inputs)
        out = self.torso(out)
        out = self.head(out)
        return out
