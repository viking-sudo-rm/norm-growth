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
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
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


def main(args):
    path = f"data/wd/wikitext-2-vaswani-lr={args.lr}-wd={args.wd}.dat"
    with open(path, "rb") as fh:
        timeseries = pickle.load(fh)
    
    ps = np.array(timeseries["pnorms"])
    ds = np.array(timeseries["dnorms"])

    mean_p = np.median(ps, axis=1)
    mean_d = np.median(ds, axis=1)
    smooth_d = savgol_filter(mean_d, 103, 3)

    # Set up phase regions.
    slope = np.sqrt(args.wd - args.wd * args.wd)
    ys = slope * mean_p
    plt.fill_between(mean_p, 0, ys, color="red", alpha=.1)
    # plt.fill_between(mean_p, ys, 1, color="blue", alpha=.1)
    plt.plot(mean_p, ys, linestyle="--", color="black")

    # See https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib.
    line, = plt.plot(mean_p, smooth_d)
    add_arrow(line)

    plt.xlabel(R"$\Vert \delta_t \Vert$")
    plt.ylabel(R"$\Vert \theta_t \Vert$")
    plt.yscale("log")
    # plt.xscale("log")

    if not os.path.exists("figs/wd"):
        os.makedirs("figs/wd")
    
    filename = f"figs/wd/lr={args.lr}-wd={args.wd}.pdf"
    plt.savefig(filename)
    print(f"Saved [green]{filename}[/green].")


if __name__ == "__main__":
    main(parse_args())
