"""Plot the learning rate vs. weight decay grid."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import argparse


SAVE = "/net/nfs.corp/allennlp/willm/cached"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", default="SGD", choices=["SGD", "AdamW"])
    parser.add_argument("--data_dir", type=str, default="grid-norms")
    return parser.parse_args()


args = parse_args()

lrs = []
wds = []
dnorms = []

data_path = f"{SAVE}/{args.data_dir}"
for name in os.listdir(data_path):
    result = re.match(r"(.*)\-lr\=(.*)\-wd\=(.*)\.dat", name)
    optim = result[1]
    lr = float(result[2])
    wd = float(result[3])

    with open(os.path.join(data_path, name), "rb") as fh:
        norms = pickle.load(fh)

    dnorm = norms[-1] - norms[0]

    if optim == args.optim:
        lrs.append(lr)
        wds.append(wd)
        # TODO: Figure this out??
        # Alternative could plot 0/1.
        log_dnorm = np.sign(dnorm) * np.log(np.abs(dnorm) + 1)
        dnorms.append(log_dnorm)

if not os.path.isdir("figs/grid"):
    os.makedirs("figs/grid")

lrs, wds, dnorms = np.array(lrs), np.array(wds), np.array(dnorms)
plt.contourf(lrs.reshape(20, 20), wds.reshape(20, 20), dnorms.reshape(20, 20), levels=2)

# sc = plt.scatter(lrs, wds, c=dnorms)
# plt.colorbar(sc)
plt.title(fR"Log norm growth with {args.optim} after $1$ epoch by $\eta, \lambda$")
plt.xlabel(R"$\eta$")
plt.ylabel(R"$\lambda$")
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"figs/grid/{args.optim}.pdf")

# contourf, pcolormesh?
