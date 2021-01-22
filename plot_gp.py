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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--window", type=int, default=100)
    return parser.parse_args()


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
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


def main(args):
    path = f"data/wd/wikitext-2-vaswani-{args.optim}-lr={args.lr}-wd={args.wd}.dat"
    with open(path, "rb") as fh:
        timeseries = pickle.load(fh)
    
    projs = np.array(timeseries["projs"])
    deltas = np.array(timeseries["dnorms"])
    deltas_sq = deltas * deltas 

    mean_proj = np.median(projs, axis=1)
    smooth_proj = smooth(mean_proj, args.window)  # SMOOTHING

    mean_delta_sq = np.median(deltas_sq, axis=1)
    mean_delta_sq = smooth(mean_delta_sq, args.window)  # SMOOTHING
    mean_delta = np.sqrt(mean_delta_sq)

    # Compute the boundary curve.
    xs = np.array(sorted(mean_delta))
    ys = -.5 * np.array(sorted(mean_delta_sq))

    # Set up scaling.
    ymin = min(min(ys), min(smooth_proj))
    ymax = max(max(ys), max(smooth_proj))
    yrange = ymax - ymin
    ymin -= yrange / 2
    ymax += yrange / 2

    # Plot all the stuff.
    plt.fill_between(xs, ymin, ys, color="red", alpha=.1)
    plt.plot(xs, ys, linestyle="--", color="black")
    line, = plt.plot(mean_delta, smooth_proj, alpha=.2)
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


if __name__ == "__main__":
    main(parse_args())
