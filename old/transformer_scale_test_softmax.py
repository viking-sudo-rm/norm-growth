from typing import Dict, List, NamedTuple
import torch
from torch.nn import Module, Linear, LayerNorm
from torch.autograd import grad
from collections import defaultdict
import matplotlib.pyplot as plt
from rich import print
from tqdm import tqdm
import pickle
import argparse
import numpy as np
from scipy.signal import medfilt
from torch.autograd import grad
from torch.nn.functional import one_hot
import torch.nn as nn
import os
from matplotlib.ticker import FormatStrFormatter

EPS = torch.finfo(torch.float32).eps

lnorm = LayerNorm(10)
lnorm.name = "LayerNorm"

TRANS_SIZE = 240

DATA = "data"
IMAGES = "figs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--font_size", type=str, default="large")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--min_h", type=int, default=12)
    parser.add_argument("--max_h", type=int, default=120)
    parser.add_argument("--step_h", type=int, default=12)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--symlog", action="store_true")
    parser.add_argument("--experiment", choices=["hid_dim", "scale"], default="hid_dim")
    return parser.parse_args()


def get_norm(network):
    params = [param.flatten() for param in network.parameters()]
    return torch.cat(params).norm(p=2)


class TransConfig(NamedTuple):
    n_model: int
    n_classes: int
    n_vocab: int = 100
    n_layers: int = 6
    n_heads: int = 12
    ff_dim: int = 50
    bias: bool = True
    residual: bool = True


class SelfAttention(Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        assert config.n_model % config.n_heads == 0
        key_dim = config.n_model // config.n_heads
        self.key_map = Linear(config.n_model, key_dim, bias=config.bias)
        self.query_map = Linear(config.n_model, key_dim, bias=config.bias)
        self.value_map = Linear(config.n_model, key_dim, bias=config.bias)

    def forward(self, inputs):
        keys = self.key_map(inputs)
        queries = self.query_map(inputs)
        values = self.value_map(inputs)
        attn_weights = torch.softmax(keys @ queries.t(), dim=-1)
        return attn_weights @ values


class PoolHeads(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList(
            [SelfAttention(config) for _ in range(config.n_heads)]
        )
        self.pooler = nn.Linear(config.n_model, config.n_model, bias=config.bias)
        self.lnorm = LayerNorm(config.n_model)

    def forward(self, inputs):
        heads = [head(inputs) for head in self.heads]
        outputs = self.pooler(torch.cat(heads, dim=-1))
        if self.config.residual:
            return self.lnorm(outputs + inputs)
        else:
            return self.lnorm(outputs)


class FullyConnected(Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        self.first = Linear(config.n_model, config.ff_dim, bias=config.bias)
        self.second = Linear(config.ff_dim, config.n_model, bias=config.bias)
        self.lnorm = LayerNorm(config.n_model)
        self.residual = config.residual

    def forward(self, inputs):
        hidden = self.first(inputs).relu()
        hidden = self.second(hidden)
        if self.residual:
            return self.lnorm(hidden + inputs)
        else:
            return self.lnorm(hidden)


class TransformerLayer(Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        self.attention = PoolHeads(config)
        self.fully_connected = FullyConnected(config)

    def forward(self, x):
        attention = self.attention(x)
        return self.fully_connected(attention)


class Embedding(Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        self.embed = torch.nn.Embedding(config.n_vocab, config.n_model)
        self.pos_embed = torch.nn.Embedding(512, config.n_model)

    def forward(self, x):
        pos = torch.arange(len(x), device=x.device)
        return self.embed(x) + self.pos_embed(pos)


class Transformer(Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        self.embed = Embedding(config)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layers)]
        )
        self.output = nn.Linear(config.n_model, config.n_classes, bias=config.bias)

        not_bias = "bias" if not config.bias else ""
        not_res = "res" if not config.residual else ""
        self.name = "Trans"
        if not_bias or not_res:
            self.name += " w/o " + ", ".join(x for x in [not_bias, not_res] if x)

    def forward(self, inputs):
        inputs = self.embed(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return self.output(inputs)


def scale_params(module, scale: int) -> List[torch.Tensor]:
    old_params = []
    for param in module.parameters():
        old_params.append(param.data.clone())
        param.data *= scale
    return old_params


def unscale_params(module, old_params: List[torch.Tensor]) -> None:
    for param, old_data in zip(module.parameters(), old_params):
        param.data = old_data


def get_networks(n_classes, scale: float = 1.0):
    networks = [
        Transformer(TransConfig(TRANS_SIZE, n_classes)),
        Transformer(TransConfig(TRANS_SIZE, n_classes, bias=False)),
        Transformer(TransConfig(TRANS_SIZE, n_classes, residual=False)),
        Transformer(TransConfig(TRANS_SIZE, n_classes, bias=False, residual=False)),
    ]

    if scale != 1.0:
        for network in networks:
            scale_params(network, scale)

    return networks


def new_tracker():
    return {
        "mean": [],
        "std": [],
    }


def get_metrics(network, criterion, token_ids, labels, one_hot_labels):
    preds = network(token_ids)
    loss = criterion(preds, labels)
    params = [p for p in network.parameters() if p.requires_grad]
    grads = grad(loss, params)

    lr = 1e-3
    exp = 0.
    for p, g in zip(params, grads):
        p = p.flatten()
        g = g.flatten()
        exp += (lr**2 * g.norm(p=2)**2 - 2 * lr * p @ g).item()

    params = torch.cat([p.flatten() for p in params])
    grads = torch.cat([g.flatten() for g in grads])
    proj = params @ grads / (params.norm(p=2) * grads.norm(p=2) + 1e-9)

    # expansion = lr * lr * grads.norm(p=2)**2 - 2 * lr * params @ grads

    return {
        # We use cosine similarity because it is normalized for the fact that loss grows with num hidden???
        "proj": proj.item(),
        "exp": exp,
        # "homo-proj": homo_projs.mean().item(),
    }


def get_metrics_by_hid_dim(args):
    metrics = {
        "exp": defaultdict(new_tracker),
        "proj": defaultdict(new_tracker),
    }
    token_ids = torch.randint(0, 100, size=[512])
    criterion = torch.nn.CrossEntropyLoss()

    hid_dims = list(range(args.min_h, args.max_h, args.step_h))
    for hid_dim in hid_dims:
        print(f"=> h={hid_dim}...")
        t = tqdm(total=args.n_trials * 4)
        data = {
            "exp": defaultdict(list),
            "proj": defaultdict(list)
        }
        for trial in range(args.n_trials):
            networks = get_networks(hid_dim, scale=args.scale)
            labels = torch.randint(0, hid_dim, size=(512,))
            one_hot_labels = one_hot(labels, hid_dim)
            for network in networks:
                mets = get_metrics(
                    network, criterion, token_ids, labels, one_hot_labels
                )
                for metric, value in mets.items():
                    data[metric][network.name].append(value)
                t.update()

        for metric in ["proj", "exp"]:
            for network in networks:
                mean = np.mean(data[metric][network.name])
                std = np.std(data[metric][network.name])
                metrics[metric][network.name]["mean"].append(mean)
                metrics[metric][network.name]["std"].append(std)

        t.close()

    return hid_dims, metrics


def get_metrics_by_scale(args):
    metrics = {
        "exp": defaultdict(new_tracker),
        "proj": defaultdict(new_tracker),
    }
    token_ids = torch.randint(0, 100, size=[512])
    criterion = torch.nn.CrossEntropyLoss()
    scales = np.linspace(1, 10, 30)
    hid_dims = [12 * 2 ** i for i in range(1, 5)]
    for hid_dim in hid_dims:
        print(f"=> h={hid_dim}...")
        t = tqdm(total=len(scales) * args.n_trials)
        data = defaultdict(list)
        for trial in range(args.n_trials):
            network = Transformer(TransConfig(TRANS_SIZE, hid_dim))
            labels = torch.randint(0, hid_dim, size=(512,))
            one_hot_labels = one_hot(labels, hid_dim)
            for scale in scales:
                old_params = scale_params(network, scale)
                mets = get_metrics(
                    network, criterion, token_ids, labels, one_hot_labels
                )
                for metric, value in mets.items():
                    data[metric, hid_dim, scale].append(value)
                unscale_params(network, old_params)
                t.update()

        for metric in metrics:
            for scale in scales:
                mean = np.mean(data[metric, hid_dim, scale])
                std = np.std(data[metric, hid_dim, scale])
                metrics[metric][hid_dim]["mean"].append(mean)
                metrics[metric][hid_dim]["std"].append(std)

        t.close()

    return scales, metrics


# Desired plots:
#   1) proj as a function of hidden_dim by architecture (fixed at init)
#   2) proj as a function of scale by hidden_dim (fixed at standard architecture)


def main_hid_dim(args):
    if not args.load:
        data = get_metrics_by_hid_dim(args)
    else:
        with open(f"{DATA}/scale-softmax-init-{args.scale}.dat", "rb") as fh:
            data = pickle.load(fh)

    with open(f"{DATA}/scale-softmax-init-{args.scale}.dat", "wb") as fh:
        pickle.dump(data, fh)

    scale = str(int(args.scale)) if args.scale != 1.0 else ""
    latex = {
        # "proj": R"$\theta_0^\top \cdot \nabla L_{| \theta_0}$",
        "proj": R"$\mathrm{cos}("
        + scale
        + R"\theta_0, \nabla L_{| "
        + scale
        + R"\theta_0})$",
        # "homo-proj": R"$f_0^\top \cdot ( \mathrm{softmax}(f_0) - y )$",
        "homo-proj": R"$\mathrm{cos}( f_{"
        + scale
        + R" \theta_0}, \mathrm{softmax}(f_{"
        + scale
        + R"\theta_0}) - y )$",
        "exp": Rf"$\Delta \Vert {scale} \theta_0 \Vert^2$"
    }

    max_mean = -float("inf")
    hid_dims, metrics = data
    for metric, net_data in metrics.items():
        plt.figure()
        for net, series in net_data.items():
            means = np.asarray(series["mean"])
            max_mean = max(max_mean, np.max(np.abs(means)))
            errors = 2 * np.asarray(series["std"])
            plt.plot(hid_dims, means, label=net)
            plt.fill_between(hid_dims, means - errors, means + errors, alpha=0.1)

        plt.legend(fontsize=args.font_size)
        plt.ticklabel_format(scilimits=[1, 3])
        plt.xlabel("num classes", fontsize=args.font_size)
        plt.ylabel(f"{latex[metric]}", fontsize=args.font_size)
        plt.ylim([-2 * max_mean, 2 * max_mean])
        if args.symlog:
            plt.yscale("symlog")
        plt.title(f"transformer variants at init by hidden dim")

        if not os.path.isdir(f"{IMAGES}/scale-softmax"):
            os.makedirs(f"{IMAGES}/scale-softmax")
        title = f"{metric}-{args.scale}"
        plt.savefig(f"{IMAGES}/scale-softmax/{title}.pdf")


def main_scale(args):
    if not args.load:
        data = get_metrics_by_scale(args)
    else:
        with open(f"{DATA}/scale-softmax-scale.dat", "rb") as fh:
            data = pickle.load(fh)

    with open(f"{DATA}/scale-softmax-scale.dat", "wb") as fh:
        pickle.dump(data, fh)

    latex = {
        # "proj": R"$\theta_0^\top \cdot \nabla L_{| \theta_0}$",
        "proj": R"$\mathrm{cos}(c\theta_0, \nabla L_{| c\theta_0})$",
        # "homo-proj": R"$f_0^\top \cdot ( \mathrm{softmax}(f_0) - y )$",
        "homo-proj": R"$\mathrm{cos}( f_{c\theta_0}, \mathrm{softmax}(f_{c\theta_0}) - y )$",
        "exp": Rf"$\Delta \Vert c \theta_0 \Vert^2$"
    }

    scales, metrics = data
    for metric, net_data in metrics.items():
        max_mean = -float("inf")
        plt.figure()
        for hid_dim, series in net_data.items():
            means = np.asarray(series["mean"])
            max_mean = max(max_mean, np.max(np.abs(means)))
            # TODO: Percentile instead?
            errors = 2 * np.asarray(series["std"])
            plt.plot(scales, means, label=f"$h={hid_dim}$")
            plt.fill_between(scales, means - errors, means + errors, alpha=0.1)
        plt.legend(fontsize=args.font_size)
        plt.ticklabel_format(scilimits=[1, 3])
        plt.xlabel("scale $c$", fontsize=args.font_size)
        plt.ylabel(f"{latex[metric]}", fontsize=args.font_size)
        plt.ylim([-2 * max_mean, 2 * max_mean])
        plt.yscale("symlog")
        plt.title(f"projection by scale for different hidden dims")

        if not os.path.isdir(f"{IMAGES}/scale-softmax"):
            os.makedirs(f"{IMAGES}/scale-softmax")
        title = f"{metric}-scales"
        plt.savefig(f"{IMAGES}/scale-softmax/{title}.pdf")


if __name__ == "__main__":
    args = parse_args()
    if args.experiment == "hid_dim":
        main_hid_dim(args)
    elif args.experiment == "scale":
        main_scale(args)
