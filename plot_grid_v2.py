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
    parser.add_argument("--norm-type", type=str, default=None, choices=["linear", "encoder"])
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

    dnorm = norms[args.norm_type][-1] - norms[args.norm_type][0]

    if optim == args.optim:
        lrs.append(lr)
        wds.append(wd)
        # TODO: Figure this out??
        # Alternative could plot 0/1.
#        log_dnorm = np.sign(dnorm) * np.log(np.abs(dnorm) + 1)
        dnorms.append(dnorm)

if not os.path.isdir("/home/vivekr/figs/grid"):
    os.makedirs("/home/vivekr/figs/grid")

lrs, wds, dnorms = np.array(lrs), np.array(wds), np.array(dnorms)

Zsquare = np.zeros((20, 20))
Xdict = {float(n): i for i, n in enumerate(np.unique(lrs))}
Ydict = {float(n): i for i, n in enumerate(np.unique(wds))}

for x, y, z in zip(lrs, wds, dnorms):
    Zsquare[Ydict[float(y)], Xdict[float(x)]] = z

cs = plt.contourf(np.unique(lrs), np.unique(wds), Zsquare, levels=[-100000.0, 0, 100000.0])
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
           for pc in cs.collections]

plt.contour(np.unique(lrs), np.unique(wds), Zsquare, levels=[-100000.0, 0, 100000.0], linestyles="-.", colors="black", linewidths=3)
if args.optim == "AdamW":
    # Plot pytorch default
    k = plt.scatter(0.001, 0.01, marker="^", color="orange", s=80, label="Default")
    proxy.append(k)
plt.legend(proxy, ["Decreasing norm", "Increasing norm", "PyTorch default"], loc="lower right", fontsize=16)
plt.show()

# sc = plt.scatter(lrs, wds, c=dnorms)
# plt.colorbar(sc)
plt.title(fR"Norm growth with {args.optim} after $1$ epoch by $\eta, \lambda$")
plt.xlabel(R"$\eta$")
plt.ylabel(R"$\lambda$")
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"/home/vivekr/figs/grid/{args.optim}-{args.norm_type}.pdf", bbox_inches="tight")

# contourf, pcolormesh?
