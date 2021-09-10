from typing import List, Type, Union

import torch as T

from anvil.models.utils import get_device


class Actor(T.nn.Module):
    """
    The actor network which approximates the Q or Value functions.

    :param encoder: the encoder network
    :param torso: the torso network
    :param heads: a list of network heads
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        head: T.nn.Module,
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

    def forward(self, input: T.Tensor) -> List[T.Tensor]:
        out = self.encoder(input)
        out = self.torso(out)
        out = self.head(out)
        return out
