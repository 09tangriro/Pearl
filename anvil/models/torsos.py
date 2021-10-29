from typing import List, Optional, Type

import torch as T


class MLP(T.nn.Module):
    def __init__(
        self, layer_sizes: List[int], activation_fn: Optional[Type[T.nn.Module]] = None
    ):
        super().__init__()

        layers = []
        for i, size in enumerate(layer_sizes[:-1]):
            layers += [T.nn.Linear(size, layer_sizes[i + 1])]
            if activation_fn:
                layers += [activation_fn()]
        self.model = T.nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)
