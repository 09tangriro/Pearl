from enum import Enum


class NetworkType(Enum):
    MLP = "mlp"
    PARAMETER = "parameter"


class TrajectoryType(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


class TrainFrequencyType(Enum):
    EPISODE = "episode"
    STEP = "step"


class GoalSelectionStrategy(Enum):
    """Goal selection strategy for HER Replay Buffer"""

    FINAL = "final"
    FUTURE = "future"
    EPISODE = "episode"
