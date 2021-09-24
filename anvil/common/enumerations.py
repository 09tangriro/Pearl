from enum import Enum


class NetworkType(Enum):
    MLP = "mlp"
    PARAMETER = "parameter"


class TrajectoryType(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
