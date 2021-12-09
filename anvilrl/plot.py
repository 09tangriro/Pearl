import math
import os
from enum import Enum
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


class Metric(Enum):
    """
    Enum for the different plots.
    """

    REWARD = "reward"
    ACTOR_LOSS = "actor"
    CRITIC_LOSS = "critic"
    DIVERGENCE = "divergence"
    ENTROPY = "entropy"


def read_tensorboard_data(
    path: str, metric: str
) -> List[event_accumulator.ScalarEvent]:
    """
    Reads tensorboard data from a given path.
    :param path: path to the tensorboard data
    :param plot: the plot to read
    """
    metric = Metric(metric.lower())
    if metric == Metric.REWARD:
        tag = "Reward/episode_reward"
    elif metric == Metric.ACTOR_LOSS:
        tag = "Loss/actor_loss"
    elif metric == Metric.CRITIC_LOSS:
        tag = "Loss/critic_loss"
    elif metric == Metric.DIVERGENCE:
        tag = "Metrics/divergence"
    elif metric == Metric.ENTROPY:
        tag = "Metrics/entropy"

    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        },
    )

    ea.Reload()
    return ea.Scalars(tag)


def get_axis_data(
    data: List[event_accumulator.ScalarEvent],
    x_axis: str = "step",
    y_axis: str = "value",
) -> Tuple[List[Any], List[Any]]:
    """
    Gets the axis data from the given data.
    :param data: the data to get the axis data from
    :param x_axis: the x axis tag
    :param y_axis: the y axis tag
    """
    if x_axis == "step":
        x_data = [event.step for event in data]
    elif x_axis == "wall_time":
        x_data = [event.wall_time for event in data]
    elif x_axis == "value":
        x_data = [event.value for event in data]

    if y_axis == "value":
        y_data = [event.value for event in data]
    elif y_axis == "wall_time":
        y_data = [event.wall_time for event in data]
    elif y_axis == "step":
        y_data = [event.step for event in data]

    return x_data, y_data


def plot(
    paths: List[str],
    metric: str,
    titles: List[str],
    num_cols: int,
    xlabel: str = "Step",
    ylabel: str = "Value",
    x_axis: str = "step",
    y_axis: str = "value",
    log_y: bool = False,
    save_types: List[str] = ["pdf"],
) -> None:
    save_path: Optional[str] = os.path.join(os.getcwd(), "plot")
    """
    Plots the given data.
    :param paths: the paths to the tensorboard data
    :param metric: the metric to plot
    :param xlabel: the x axis tag
    :param ylabel: the y axis tag
    :param x_axis: the x axis data
    :param y_axis: the y axis data
    :param log_y: whether to log the y axis
    :param save_types: the save types
    :param save_path: where to save the plots
    """
    num_plots = len(paths)
    num_rows = math.ceil(num_plots / num_cols)
    plt.figure

    for i, path in enumerate(paths):
        data = read_tensorboard_data(path, metric)
        x_data, y_data = get_axis_data(data, x_axis, y_axis)

        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(x_data, y_data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titles[i])
        plt.grid(True)
        if log_y:
            plt.yscale("log")
        plt.tight_layout()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for save_type in save_types:
            plt.savefig(f"{save_path}.{save_type}")

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--paths", nargs="+", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--titles", nargs="+", type=str, required=True)
    parser.add_argument("--num-cols", type=int, required=True)
    parser.add_argument("--xlabel", type=str, default="Step")
    parser.add_argument("--ylabel", type=str, default="Value")
    parser.add_argument("--x-axis", type=str, default="step")
    parser.add_argument("--y-axis", type=str, default="value")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--save-types", nargs="+", type=str, default=["pdf"])
    parser.add_argument("--save-path", type=str, default=None)

    args = parser.parse_args()

    plot(
        args.paths,
        args.metric,
        args.titles,
        args.num_cols,
        args.xlabel,
        args.ylabel,
        args.x_axis,
        args.y_axis,
        args.log_y,
        args.save_types,
        args.save_path,
    )
