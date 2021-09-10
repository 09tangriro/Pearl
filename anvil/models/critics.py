from typing import List, Union

import torch as T

from anvil.models.utils import get_device


class Critic(T.nn.Module):
    """
    The critic network which approximates the Q or Value functions. Define the heads as a
    list to allow for potential multiple head output networks.

    :param encoder: the encoder network
    :param torso: the torso network
    :param heads: a list of network heads
    """

    def __init__(
        self,
        encoder: T.nn.Module,
        torso: T.nn.Module,
        heads: List[T.nn.Module],
        device: Union[T.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.encoder = encoder.to(device)
        self.torso = torso.to(device)
        self.heads = [head.to(device) for head in heads]

    def forward(self, input: T.Tensor) -> List[T.Tensor]:
        out = self.encoder(input)
        out = self.torso(out)
        out = [head(out) for head in self.heads]
        return out
