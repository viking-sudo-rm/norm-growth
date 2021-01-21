import torch
from torch import nn
from torch.autograd import grad
from rich import print
from collections import defaultdict
from torch.optim import AdamW
import tqdm
import argparse
import pickle

from src.saturate import saturate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--n_batch", type=int, default=5)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--x_dim", type=int, default=50)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--y_dim", type=int, default=1)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save", type=str, default="data/time_series.dat")
    parser.add_argument("--bce", action="store_true")
    return parser.parse_args()


def ntk(net, x1, x2=None):
    """Compute the NTK matrix comparing `x1` against `x2`.

    The number of gradient computations (the major bottleneck) is `len(x1) + len(x2)`.
    """
    # kernel = torch.empty(len(x1), len(x2), device=x1.device)

    # Compute the gradients with respect to the first list of inputs.
    grad1 = [
        torch.cat([x.flatten() for x in grad(net(xi), net.parameters())]) for xi in x1
    ]
    grad1 = torch.stack(grad1, dim=0)

    # Compute the gradients with respect to the second list of inputs.
    if x2 is None:
        grad2 = grad1.T
    else:
        grad2 = [
            torch.cat([x.flatten() for x in grad(net(xj), net.parameters())])
            for xj in x2
        ]
        grad2 = torch.stack(grad2, dim=-1)

    return grad1 @ grad2


def linearize(f, X, Y):
    def _f(x):
        # We can't memoize this kernel, since the parameters in f can change.
        # FIXME: Rarely (but sometimes) get an error computing the true inverse here.
        return f(x) - ntk(f, x, X) @ ntk(f, X, X).pinverse() @ (f(X) - Y)

    return _f


def solve_params(f, X, Y):
    """Returns `omega_t`, the difference in parameters updated by linear dynamics.

    See: https://arxiv.org/pdf/1902.06720.pdf
    """
    # TODO: Can actually memoize this gradient computation from the ntk.
    grads = [grad(f(xi), f.parameters()) for xi in X]
    # import pdb; pdb.set_trace()
    grads_trans = [torch.stack(grad, dim=-1) for grad in zip(*grads)]
    memo = ntk(f, X).pinverse() @ (Y - f(X))
    memo = memo.squeeze(dim=1)
    return [grad_trans @ memo for grad_trans in grads_trans]


def metrics(preds, labels):
    # The train loss should be 0 based on Vivek's derivation.
    return {
        "loss": (preds - labels).norm(p=2).item() / len(preds),
        "saturation": (2 * preds - 1).abs().mean().item(),
    }


def map_append(time_series, metrics):
    for name, value in metrics.items():
        time_series[name].append(value)


def make_data(args):
    """Create linearly separable data."""
    print("Generating data..")
    n_test = args.n_train // 10
    W = torch.randn(args.x_dim, args.y_dim, device=args.cuda)
    X = torch.randn(args.n_train, args.x_dim, device=args.cuda)
    Y = ((X @ W) > 0).float()
    x = torch.randn(n_test, args.x_dim, device=args.cuda)
    y = ((x @ W) > 0).float()
    return X, Y, x, y


def make_network(args):
    print("Creating network..")
    net = torch.nn.Sequential(
        torch.nn.Linear(args.x_dim, args.h_dim, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(args.h_dim, args.y_dim, bias=False),
        torch.nn.Sigmoid(),
    )
    # net = torch.nn.Sequential(
    #     torch.nn.Linear(args.x_dim, args.h_dim),
    #     torch.nn.Sigmoid(),
    #     torch.nn.Linear(args.h_dim, args.y_dim),
    #     torch.nn.Sigmoid(),
    # )
    if args.cuda is not None:
        net.cuda(args.cuda)
    return net


def param_space_linearize(f, X, Y, x, y):
    # # This part takes the prediction points.
    # order0 = f(X).squeeze(dim=-1)
    # order1 = torch.stack(
    #     [torch.cat([g.flatten() for g in grad(f(xj), f.parameters())]) for xj in X],
    #     dim=0,
    # )
    # with saturate(f, no_grad=False):
    #     sat_order0 = f(X).squeeze(dim=-1)
    #     sat_order1 = torch.stack(
    #         [torch.cat([g.flatten() for g in grad(f(xj), f.parameters())]) for xj in X],
    #         dim=0,
    #     )

    # # This part takes the train data.
    omegas = solve_params(f, X, Y)
    # flat_omegas = torch.cat([omega.flatten() for omega in omegas])
    # f_unsat = order0 + order1 @ flat_omegas
    # f_sat = sat_order0 + sat_order1 @ flat_omegas

    train = f(X)
    test = f(x)
    with saturate(f):
        sat_train = f(X)
        sat_test = f(x)

    print(
        "init-round",
        {
            "train_acc": (train.round() == Y).float().mean(),
            "test_acc": (test.round() == y).float().mean(),
        },
    )
    print(
        "init-sat",
        {
            "train_acc": (sat_train == Y).float().mean(),
            "test_acc": (sat_test == y).float().mean(),
        },
    )

    for param, omega in zip(f.parameters(), omegas):
        param.data += omega

    train = f(X)
    test = f(x)
    with saturate(f):
        sat_train = f(X)
        sat_test = f(x)

    print(
        "fine-round",
        {
            "train_acc": (train.round() == Y).float().mean(),
            "test_acc": (test.round() == y).float().mean(),
        },
    )
    print(
        "fine-sat",
        {
            "train_acc": (sat_train == Y).float().mean(),
            "test_acc": (sat_test == y).float().mean(),
        },
    )
    # We expect normal/saturated to have the same performance, because the toy network is scale-invariant.


def run_experiment(args):
    X, Y, x, y = make_data(args)
    f = make_network(args)
    f_lin = linearize(f, X, Y)

    param_space_linearize(f, X, Y, x, y)

    time_series = {
        "train": defaultdict(list),
        "test": defaultdict(list),
        "train+NTK": defaultdict(list),
        "test+NTK": defaultdict(list),
    }
    with torch.no_grad():
        map_append(time_series["train"], metrics(f(X), Y))
        map_append(time_series["test"], metrics(f(x), y))
    map_append(time_series["train+NTK"], metrics(f_lin(X), Y))
    map_append(time_series["test+NTK"], metrics(f_lin(x), y))

    optim = AdamW(f.parameters())
    # See https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html.
    criterion = nn.BCELoss() if args.bce else nn.MSELoss()
    print("Starting optimization...")
    for epoch in range(args.n_epochs):
        print(f"  => epoch={epoch}")
        # Create a shuffled copy of the data. We need to preserve the original order for NTK.
        perm = torch.randperm(len(X))
        X_perm = X[perm, :]
        Y_perm = Y[perm, :]

        for batch in tqdm.trange(args.n_train // args.n_batch):
            batch_X = X_perm[batch * args.n_batch : (batch + 1) * args.n_batch]
            batch_Y = Y_perm[batch * args.n_batch : (batch + 1) * args.n_batch]
            preds = f(batch_X)
            optim.zero_grad()
            loss = criterion(preds, batch_Y)
            loss.backward()
            optim.step()

        # Record metrics at end of epoch.
        with torch.no_grad():
            map_append(time_series["train"], metrics(f(X), Y))
            map_append(time_series["test"], metrics(f(x), y))
        map_append(time_series["train+NTK"], metrics(f_lin(X), Y))
        map_append(time_series["test+NTK"], metrics(f_lin(x), y))

    return time_series


def main(args):
    if args.load is None:
        time_series = run_experiment(args)
        with open(args.save, "wb") as fh:
            pickle.dump(time_series, fh)
    else:
        with open(args.load, "rb") as fh:
            time_series = pickle.load(fh)

    import matplotlib.pyplot as plt

    styles = [".-", "--", "-", ":"]

    metric_names = time_series["train"].keys()
    for metric in metric_names:
        plt.figure()
        for style, (key, data) in zip(styles, time_series.items()):
            plt.plot(data[metric], label=key, linestyle=style)
        plt.legend()
        plt.title(f"{metric} over training")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        path = f"figs/toy-ntk/{metric}.pdf"
        plt.savefig(path)
        print(f"=> Saved {path}.")


if __name__ == "__main__":
    main(parse_args())
