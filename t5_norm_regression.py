import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
from rich import print
import os

DATA = os.getenv("DATA")
PATH = f"{DATA}/bsl/t5-deriv"
FIG_PATH = "figs/t5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf")
    parser.add_argument("--small_font", type=int, default=14)
    parser.add_argument("--large_font", type=int, default=18)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("--min", action="store_true")
    return parser.parse_args()


args = parse_args()
PATH += f"/min-{args.n}" if args.min else f"/norm-{args.n}"

with open(f"{PATH}/norms.dat", "rb") as fh:
    y = pickle.load(fh)
    y = np.array(y).reshape(-1, 1)

with open(f"{PATH}/ckpts.dat", "rb") as fh:
    x = pickle.load(fh)
    x = np.array(x).reshape(-1, 1)

def regress(x, y):
    reg = LinearRegression()
    reg.fit(x, y)
    return reg.score(x, y), reg

lin_r2, lin_reg = regress(x, y)
sqrt_r2, sqrt_reg = regress(np.sqrt(x), y)
exp_r2, exp_reg = regress(np.exp(-x), y)
root4_r2, root4_reg = regress(np.sqrt(np.sqrt(x)), y)

print("lin", lin_r2)
print("sqrt", sqrt_r2)
print("exp", exp_r2)
print("root4", root4_r2)

import matplotlib.pyplot as plt
times = {"fontname": "Times"}

if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

plt.plot(x, y, ".", label="$\\rho(t)$")
# plt.plot(x, lin_reg.predict(x), label="$\\hat \\rho(t) = at + b$")
plt.plot(x, sqrt_reg.predict(np.sqrt(x)), label="$\\hat \\rho(t) = a\\sqrt{t} + b$")
# plt.plot(x, sqrt_reg.predict(np.exp(-x)), label="$\\hat \\rho(t) = a\\exp(-t) + b$")
plt.legend(prop={"size": args.small_font})
# plt.title("Param growth with $\\sqrt{\cdot}$ fit", fontsize=args.large_font)
plt.xlabel("Checkpoint $t$", fontsize=args.small_font)
plt.ylabel(R"Min parameter magnitude" if args.min else "Parameter norm", fontsize=args.small_font)
plt.tight_layout()
path = f"{FIG_PATH}/t5-norm.{args.format}"
plt.savefig(path)
print(f"[green]=>[/green] Saved fig to {path}.")

with open(f"{PATH}/norms_by_layer.dat", "rb") as fh:
    data = pickle.load(fh)

hsv = plt.get_cmap('hsv')
colors = hsv(np.linspace(0, 0.8, 12))
plt.figure()
layers = reversed(range(12))
for color, layer in zip(colors, layers):
    y = np.array(data[layer]).reshape(-1, 1)
    r2, reg = regress(np.sqrt(x), y)
    plt.plot(x, y, ".", color=color, label=f"Layer {layer + 1}")
    # plt.plot(x, reg.predict(np.sqrt(x)), color=color, label=f"Layer {layer + 1}")
plt.legend(prop={"size": args.small_font})
plt.xlabel("Checkpoint $t$", fontsize=args.small_font)
plt.ylabel("Min parameter magnitude" if args.min else "Parameter norm", fontsize=args.small_font)
# plt.title("Param growth with $\\sqrt{\cdot}$ fit by layer", fontsize=args.large_font)
plt.tight_layout()
path = f"{FIG_PATH}/t5-norm-by-layer.{args.format}"
plt.savefig(path)
print(f"[green]=>[/green] Saved fig to {path}.")

with open(f"{PATH}/dir_sims.dat", "rb") as fh:
    dir_sims = pickle.load(fh)

plt.figure()
plt.plot(x[1:], dir_sims, ".")
# plt.title(R"Param directional similarity", fontsize=args.large_font)
plt.xlabel("Checkpoint $t$", fontsize=args.small_font)
plt.ylabel(R"$\mathrm{cos}(\theta_t, \theta_{t+1})$", fontsize=args.small_font)
plt.tight_layout()
path = f"{FIG_PATH}/t5-dir.{args.format}"
plt.savefig(path)
print(f"[green]=>[/green] Saved fig to {path}.")

with open(f"{PATH}/dir_sims_by_layer.dat", "rb") as fh:
    dir_sims_by_layer = pickle.load(fh)

plt.figure()
layers = reversed(range(12))
for color, layer in zip(colors, layers):
    y = np.array(dir_sims_by_layer[layer]).reshape(-1, 1)
    plt.plot(x[1:], y, ".", color=color, label=f"Layer {layer + 1}")
plt.legend(prop={"size": args.small_font})
# plt.title(R"Param directional similarity by layer", fontsize=args.large_font)
plt.xlabel("Checkpoint $t$", fontsize=args.small_font)
plt.ylabel(R"$\mathrm{cos}(\theta_t, \theta_{t+1})$", fontsize=args.small_font)
plt.tight_layout()
path = f"{FIG_PATH}/t5-dir-by-layer.{args.format}"
plt.savefig(path)
print(f"[green]=>[/green] Saved fig to {path}.")
