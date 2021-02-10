"""Plot the network trajectory in the gradient norm / parameter norm plane.

This relates to theory derived in the paper about the effect of weight decay on norm growth.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from scipy.signal import savgol_filter
from rich import print
import matplotlib.animation as animation


class experiment:
    default = None
    registry = {}

    def __init__(self, name=None, default=False):
        self.name = name
        self.default = default

    def __call__(self, callback):
        name = self.name or callback.__name__
        self.registry[name] = callback
        if self.default:
            type(self).default = name
        return callback
    
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--exp", type=str, default=cls.default, choices=list(cls.registry))
    
    @classmethod
    def run(cls, args):
        cls.registry[args.exp](args)


def parse_args():
    parser = argparse.ArgumentParser()
    experiment.add_arguments(parser)
    parser.add_argument("--optim", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="cache")
    return parser.parse_args()


def add_arrow(line, position=None, direction='right', size=15, color=None):
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def smooth(signal, window=100):
    return np.convolve(signal, np.ones(window) / window, mode="valid")


def consume(timeseries, args):
    """Get parameter time series data from `timeseries`."""
    projs = np.array(timeseries["projs"])
    deltas = np.array(timeseries["dnorms"])
    deltas_sq = deltas * deltas 

    # TODO: What is the right statistic to use here? Median seems the best.
    mean_proj = np.median(projs, axis=1)
    mean_delta_sq = np.median(deltas_sq, axis=1)

    # Smooth these things.
    mean_proj = smooth(mean_proj, args.window)  # SMOOTHING
    mean_delta_sq = smooth(mean_delta_sq, args.window)  # SMOOTHING

    mean_delta = np.sqrt(mean_delta_sq)

    return mean_delta, mean_proj


@experiment(default=True)
def main(args):
    path = f"{args.data_dir}/wd/wikitext-2-vaswani-{args.optim}-lr={args.lr}-wd={args.wd}.dat"
    with open(path, "rb") as fh:
        timeseries = pickle.load(fh)

    mean_delta, mean_proj = consume(timeseries, args)

    # Compute the boundary curve.
    xmax = max(mean_delta)
    xs = np.linspace(0, xmax, 1000)
    ys = -.5 * xs * xs

    # Set up scaling.
    ymin = min(min(ys), min(mean_proj))
    ymax = max(max(ys), max(mean_proj))
    yrange = ymax - ymin
    ymin -= yrange / 2
    ymax += yrange / 2

    # Plot all the stuff.
    plt.fill_between(xs, ymin, ys, color="red", alpha=.1)
    plt.plot(xs, ys, linestyle="--", color="black")
    line, = plt.plot(mean_delta, mean_proj)
    add_arrow(line)

    # Add various labels.
    plt.title(fR"Trajectory with {args.optim} ($\eta = {args.lr}, \lambda = {args.wd}$)")
    plt.xlabel(R"$\Vert \delta_t \Vert$")
    plt.ylabel(R"$\theta_t^\top \cdot \delta_t$")
    plt.ylim(ymin=ymin, ymax=ymax)

    if not os.path.exists("figs/wd"):
        os.makedirs("figs/wd")

    filename = f"figs/wd/{args.optim}-lr={args.lr}-wd={args.wd}.pdf"
    plt.savefig(filename)
    print(f"Saved [green]{filename}[/green].")


@experiment()
def vary_lr(args):
    all_xs = {}
    all_ys = {}
    xmax = 0.
    ymin, ymax = float("inf"), 0.

    lrs = [1e-3, 1e-5]
    # lrs = [1e-2, 1e-3, 1e-5]
    for lr in lrs:
        path = f"{args.data_dir}/wd/wikitext-2-vaswani-{args.optim}-lr={lr}-wd={args.wd}.dat"
        with open(path, "rb") as fh:
            timeseries = pickle.load(fh)

        mean_delta, mean_proj = consume(timeseries, args)

        all_xs[lr] = mean_delta
        all_ys[lr] = mean_proj
        xmax = max(xmax, max(mean_delta))
        ymin = min(ymin, min(mean_proj))
        ymax = max(ymax, max(mean_proj))

    xs = np.linspace(0, xmax, 1000)
    ys = -.5 * xs * xs
    ymin = min(ymin, min(ys))
    ymax = max(ymax, max(ys))
    yrange = ymax - ymin
    ymin -= yrange / 2
    ymax += yrange / 2

    plt.fill_between(xs, ymin * np.ones_like(ys), ys, color="red", alpha=.1)
    plt.plot(xs, ys, linestyle="--", color="black")

    for lr in all_xs.keys():
        line, = plt.plot(all_xs[lr], all_ys[lr], label=fR"$\eta = {lr}$")
        add_arrow(line)

    # Add various labels.
    plt.legend()
    plt.xscale("log")
    plt.yscale("symlog")
    plt.title(fR"Trajectory with {args.optim} ($\lambda = {args.wd}$)")
    plt.xlabel(R"$\Vert \delta_t \Vert$")
    plt.ylabel(R"$\theta_t^\top \cdot \delta_t$")
    plt.ylim(ymin=ymin, ymax=ymax)

    if not os.path.exists("figs/wd"):
        os.makedirs("figs/wd")

    filename = f"figs/wd/{args.optim}-lr=vary-wd={args.wd}.pdf"
    plt.savefig(filename)
    print(f"Saved [green]{filename}[/green].")


@experiment()
def vary_wd(args):
    all_xs = {}
    all_ys = {}
    xmax = 0.
    ymin, ymax = float("inf"), 0.

    for wd in [1e-2, 1e-3, 1e-4]:
        path = f"{args.data_dir}/wd/wikitext-2-vaswani-{args.optim}-lr={args.lr}-wd={wd}.dat"
        with open(path, "rb") as fh:
            timeseries = pickle.load(fh)

        mean_delta, mean_proj = consume(timeseries, args)

        all_xs[wd] = mean_delta
        all_ys[wd] = mean_proj
        xmax = max(xmax, max(mean_delta))
        ymin = min(ymin, min(mean_proj))
        ymax = max(ymax, max(mean_proj))

    xs = np.linspace(0, xmax, 1000)
    ys = -.5 * xs * xs
    ymin = min(ymin, min(ys))
    ymax = max(ymax, max(ys))
    yrange = ymax - ymin
    ymin -= yrange / 2
    ymax += yrange / 2

    plt.fill_between(xs, ymin * np.ones_like(ys), ys, color="red", alpha=.1)
    plt.plot(xs, ys, linestyle="--", color="black")

    for wd in all_xs.keys():
        line, = plt.plot(all_xs[wd], all_ys[wd], label=fR"$\lambda = {wd}$")
        add_arrow(line)

    # Add various labels.
    plt.legend()
    plt.xscale("log")
    # plt.yscale("symlog")
    plt.title(fR"Trajectory with {args.optim} ($\eta = {args.lr}$)")
    plt.xlabel(R"$\Vert \delta_t \Vert$")
    plt.ylabel(R"$\theta_t^\top \cdot \delta_t$")
    plt.ylim(ymin=ymin, ymax=ymax)

    if not os.path.exists("figs/wd"):
        os.makedirs("figs/wd")

    filename = f"figs/wd/{args.optim}-lr={args.lr}-wd=vary.pdf"
    plt.savefig(filename)
    print(f"Saved [green]{filename}[/green].")


if __name__ == "__main__":
    experiment.run(parse_args())
