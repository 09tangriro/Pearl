import os
from enum import Enum
from typing import Any, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
) -> Tuple[np.ndarray, np.ndarray]:
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

    return np.array(x_data), np.array(y_data)


def get_file_name(path: str) -> str:
    """
    Gets the file name from the given path.
    :param path: the path to the file
    """
    return os.path.basename(path).split(".")[0]


def smooth(vals: np.ndarray, window: int) -> np.ndarray:
    """Smooths values using a sliding window."""

    if window > 1:
        if window > len(vals):
            window = len(vals)
        y = np.ones(window)
        x = vals
        z = np.ones(len(vals))
        mode = "same"
        vals = np.convolve(x, y, mode) / np.convolve(z, y, mode)

    return vals


def stats(
    time_series: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract rolling statistics over the common range of x values.

    :param time_series: the time series to extract the rolling statistics from
    :param window: the window size
    """
    windowed_series = np.lib.stride_tricks.sliding_window_view(time_series, window)
    std = np.std(windowed_series, axis=1)
    min = np.min(windowed_series, axis=1)
    max = np.max(windowed_series, axis=1)
    std = np.concatenate((std, np.std(windowed_series[-window + 1 :], axis=1)))
    min = np.concatenate((min, np.min(windowed_series[-window + 1 :], axis=1)))
    max = np.concatenate((max, np.max(windowed_series[-window + 1 :], axis=1)))
    return std, min, max


def plot(
    paths: List[List[str]],
    metric: str,
    titles: List[str],
    num_cols: int,
    interval: str = "bounds",
    legend: Optional[List[str]] = None,
    window: int = 0,
    xlabel: str = "Step",
    ylabel: str = "Reward",
    x_axis: str = "step",
    y_axis: str = "value",
    log_y: bool = False,
    save_types: List[str] = ["pdf"],
    save_path: Optional[str] = os.path.join(os.getcwd(), "plots/plot"),
) -> None:
    """
    Plots the given data.
    :param paths: the paths to the tensorboard data
    :param metric: the metric to plot
    :param titles: the titles of the plots
    :param num_cols: the number of columns to use
    :param interval: the interval to use
    :param legend: the legend to use
    :param window: the window size
    :param xlabel: the x axis tag
    :param ylabel: the y axis tag
    :param x_axis: the x axis data
    :param y_axis: the y axis data
    :param log_y: whether to log the y axis
    :param save_types: the save types
    :param save_path: where to save the plots
    """
    num_plots = len(paths)
    num_rows = int(np.ceil(num_plots / num_cols))
    plt.ion()
    fig = plt.figure(figsize=(num_cols * 6, num_rows * 5))
    fig.clear()
    grid = mpl.gridspec.GridSpec(
        nrows=num_rows + 1,
        ncols=1 + num_cols,
        hspace=0.3,
        height_ratios=[1] * num_rows + [0.1],
        width_ratios=[0] + [1] * num_cols,
    )
    axes = []
    for i in range(num_plots):
        ax = fig.add_subplot(grid[i // num_cols, 1 + i % num_cols])
        axes.append(ax)

    for collection, ax, title in zip(paths, axes, titles):
        agents = [get_file_name(path) for path in collection]
        num_agents = len(agents)

        if num_agents <= 10:
            cmap = "tab10"
        elif num_agents <= 20:
            cmap = "tab20"
        else:
            cmap = "rainbow"
        cmap = plt.get_cmap(cmap)

        if isinstance(cmap, mpl.colors.ListedColormap):
            colors = cmap(range(num_agents))
        else:
            colors = list(cmap(np.linspace(0, 1, num_agents)))
        agent_colors = {a: c for a, c in zip(agents, colors)}

        for path, agent in zip(collection, agents):
            data = read_tensorboard_data(path, metric)
            x_data, y_data = get_axis_data(data, x_axis, y_axis)
            mean = smooth(y_data, window=window)
            label = agent if legend is None else legend[agents.index(agent)]
            ax.plot(x_data, mean, label=label, color=agent_colors[agent], alpha=1, lw=2)
            if window > 0:
                std, min_, max_ = stats(y_data, window)
                if interval in ["std", "bounds"]:
                    if interval == "std":
                        ax.fill_between(
                            x_data,
                            mean - std,
                            mean + std,
                            alpha=0.1,
                            color=agent_colors[agent],
                            lw=0,
                        )
                    elif interval == "bounds":
                        ax.fill_between(
                            x_data,
                            min_,
                            max_,
                            alpha=0.1,
                            color=agent_colors[agent],
                            lw=0,
                        )

        # Finalize the figures.
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", tight=True, nbins=6)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: f"{x:,g}")
        )
        low, high = ax.get_xlim()
        if max(abs(low), abs(high)) >= 1e3:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.xaxis.grid(linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        if log_y:
            ax.set_yscale("log")

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for save_type in save_types:
            plt.savefig(f"{save_path}.{save_type}", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--paths", nargs="+", type=str, action="append", required=True
    )
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--titles", nargs="+", type=str, required=True)
    parser.add_argument("--num-cols", type=int, default=1)
    parser.add_argument("--interval", type=str, default="bounds")
    parser.add_argument("--legend", nargs="+", type=str)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--xlabel", type=str, default="Step")
    parser.add_argument("--ylabel", type=str, default="Reward")
    parser.add_argument("--x-axis", type=str, default="step")
    parser.add_argument("--y-axis", type=str, default="value")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--save-types", nargs="+", type=str, default=["pdf"])
    parser.add_argument(
        "--save-path", type=str, default=os.path.join(os.getcwd(), "plots/plot")
    )

    args = parser.parse_args()

    plot(
        args.paths,
        args.metric,
        args.titles,
        args.num_cols,
        args.interval,
        args.legend,
        args.window,
        args.xlabel,
        args.ylabel,
        args.x_axis,
        args.y_axis,
        args.log_y,
        args.save_types,
        args.save_path,
    )
