import os
import pickle
import matplotlib.pyplot as plt
import argparse

MODELS = os.getenv("MODELS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", choices=["adamw", "adam", "sgd"], default="sgd")
    return parser.parse_args()


def load_data(optim, sched):
    path = os.path.join(MODELS, f"finetune-trans/penn/vaswani-{optim}-{sched}-batch_data.dat")
    with open(path, "rb") as fh:
        return pickle.load(fh)

args = parse_args()
constant = load_data(args.optim, "constant_lr")
sqrt = load_data(args.optim, "sqrt_lr")
linear = load_data(args.optim, "linear_lr")

fig_dir = "figs/lr-norm"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

for key in ["norm", "lr"]:
    plt.figure()
    plt.plot(constant["step"], constant[key], label="constant")
    # plt.plot(sqrt[key], label="sqrt")
    # plt.plot(linear[key], label="linear")
    plt.xlabel("checkpoint")
    plt.ylabel(key)
    plt.legend()
    plt.title(f"{key} with {args.optim} optimizer")
    # plt.axvline(x=1000, color="black", linestyle="dotted")
    # plt.yscale("log")
    plt.savefig(os.path.join(fig_dir, f"{key}.pdf"))
