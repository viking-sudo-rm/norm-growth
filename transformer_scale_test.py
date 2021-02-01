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

EPS = torch.finfo(torch.float32).eps

lnorm = LayerNorm(10)
lnorm.name = "LayerNorm"


def get_norm(network):
    params = [param.flatten() for param in network.parameters()]
    return torch.cat(params).norm(p=2)


class SelfAttention(Module):
    def __init__(self, input_dim=10, key_dim=10, bias=True):
        super().__init__()
        self.bias = bias
        self.key_map = Linear(input_dim, key_dim, bias=bias)
        self.query_map = Linear(input_dim, key_dim, bias=bias)
        self.value_map = Linear(input_dim, input_dim, bias=bias)
        self.lnorm = LayerNorm(input_dim)

        self.name = type(self).__name__
        if not bias:
            self.name += " (no biases)"

    def forward(self, inputs):
        keys = self.key_map(inputs)
        queries = self.query_map(inputs)
        values = self.value_map(inputs)
        attn_weights = torch.softmax(keys @ queries.t(), dim=-1)
        outputs = attn_weights @ values
        return self.lnorm(outputs + inputs)


class FullyConnected(Module):
    def __init__(self, input_dim=10, hidden_dim=100, bias=True, residual=True):
        super().__init__()
        self.first = Linear(input_dim, hidden_dim, bias=bias)
        self.second = Linear(hidden_dim, input_dim, bias=bias)
        self.lnorm = LayerNorm(input_dim)
        self.residual = residual

        self.name = type(self).__name__
        if not bias and not residual:
            self.name += " (no biases or res)"
        elif not bias:
            self.name += " (no biases)"

    def forward(self, inputs):
        hidden = self.first(inputs).relu()
        hidden = self.second(hidden)
        if self.residual:
            return self.lnorm(hidden + inputs)
        else:
            return self.lnorm(hidden)


class TransformerLayer(Module):
    def __init__(
        self, input_dim=10, hidden_dim=100, key_dim=10, bias=True, residual=True
    ):
        super().__init__()
        self.attention = SelfAttention(input_dim, key_dim, bias)
        self.fully_connected = FullyConnected(input_dim, hidden_dim, bias, residual)

    def forward(self, x):
        attention = self.attention(x)
        return self.fully_connected(attention)


class Transformer(Module):
    def __init__(
        self,
        n_layers: int = 10,
        input_dim=10,
        hidden_dim=100,
        key_dim=10,
        bias=True,
        residual=True,
    ):
        super().__init__()
        self.name = f"Transformer ({n_layers})"
        self.layers = [
            TransformerLayer(input_dim, hidden_dim, key_dim, bias, residual)
            for _ in range(n_layers)
        ]

        for i, layer in enumerate(self.layers):
            # self.add_module(f"layer_{i}", layer)
            setattr(self, f"layer_{i}", layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def scale_params(module, scale: int):
    old_params = []
    for param in module.parameters():
        old_params.append(param.data.clone())
        param.data *= scale
    return old_params


def unscale_params(module, old_params):
    for param, old_data in zip(module.parameters(), old_params):
        param.data = old_data


def generate(networks, args, dim):
    inputs = torch.randn(dim, 10)
    scales = list(range(1, 50, 1)) + [1000]
    grad_projs = defaultdict(list)
    grad_projs_unnorm = defaultdict(list)

    n_samples = args.dim

    criterion = torch.nn.CrossEntropyLoss()

    norms = {network.name: get_norm(network) for network in networks}
    for network in networks:
        # outputs = []
        # all_grads = []
        print(f"[gray]=>[/gray] [green]Computing values for {network}[/green]")
        
        for acc in tqdm(np.linspace(args.acc_min, 1., args.acc_step)):
            n_mistakes = int(round(n_samples * (1 - acc)))
            for scale in scales:
                old_params = scale_params(network, scale)

                if args.cuda:
                    inputs = inputs.cuda(0)

                # Assume that the input is 1-homogeneous.
                output = network(scale * inputs)

                # Fake loss function for computing grads with softmax cross-entropy loss.
                labels = output.argmax(dim=-1)
                labels[:n_mistakes] = 9 - labels[:n_mistakes]
                # labels = torch.zeros_like(labels, dtype=torch.long)
                loss = criterion(output, labels)
                params = list(network.parameters())
                grads = torch.cat([gr.flatten() for gr in grad(loss, params)])
                params = torch.cat([param.flatten() for param in params])

                grad_projs[network.name, n_mistakes].append(
                    (params @ grads / (params.norm() * grads.norm() + EPS)).item()
                )
                grad_projs_unnorm[network.name, n_mistakes].append(
                    (params @ grads).item()
                )

                unscale_params(network, old_params)


    return {
        "scales": scales,
        "grad_projs": grad_projs,
        "grad_projs_unnorm": grad_projs_unnorm,
        "norms": norms
    }


# @torch.no_grad()
def main(args):
    if not args.load:
        networks = [
            # lnorm,
            SelfAttention(10, 10),
            SelfAttention(10, 10, bias=False),
            FullyConnected(10, 10),
            FullyConnected(10, 10, bias=False),
            FullyConnected(10, 10, bias=False, residual=False),
        ]
        data_dict = generate(networks, args, args.dim)
    else:
        with open("data/data-dict.dat", "rb") as fh:
            data_dict = pickle.load(fh)

    scales = data_dict["scales"]
    grad_projs = data_dict["grad_projs"]
    grad_projs_unnorm = data_dict["grad_projs_unnorm"]
    norms = data_dict["norms"]
    n_samples = args.dim

    plot_grad_proj(
        args, scales, grad_projs_unnorm, norms, n_mistakes=0, n_samples=n_samples, legend=True
    )

    plt.figure()
    perc_mistakes = 0.2
    n_mistakes = min(
        grad_projs.keys(),
        key=lambda tup: abs(tup[1] - round(perc_mistakes * n_samples)),
    )[1]
    plt.figure()
    plot_grad_proj(
        args, scales, grad_projs_unnorm, norms, n_mistakes=n_mistakes, n_samples=n_samples,
    )

    plt.figure()
    n_mistakes_reflection_point(args, scales, grad_projs_unnorm, norms, n_samples=n_samples)

    if not args.load:
        with open("data/data-dict.dat", "wb") as fh:
            pickle.dump(data_dict, fh)


def n_mistakes_reflection_point(
    args, scales, grad_projs, norms, n_samples: int, legend: bool = False
):
    data = defaultdict(list)
    for (name, n_mistakes), projs in grad_projs.items():
        if n_mistakes == 0:  # Prevent having infinity/etc. in plot.
            continue
        checker = [idx for idx, x in enumerate(projs) if x > 0]
        crossing_point = checker[0] if len(checker) > 0 else None
        assert crossing_point is not None
        acc = (n_samples - n_mistakes) / n_samples
        # TODO: Changed this. Make sure it's right.
        crossing_norm = scales[crossing_point] * norms[name]
        data[name].append((acc, crossing_norm))

    for name in data:
        plt.plot(*zip(*sorted(data[name])), label=name)

    if legend:
        plt.legend(fontsize=args.font_size)
    plt.xlabel("Accuracy", fontsize=args.font_size)
    plt.ylabel("Norm with projection equilibrium", fontsize=args.font_size)
    plt.title("Norm with proj equilibrium")
    path = f"images/grad-proj/full.{args.format}"
    plt.savefig(path)
    print("Saved", path)


def main_depth(args):
    networks = [Transformer(n_layers=depth) for depth in range(1, args.max_depth)]
    
    if args.load:
        with open("data/depth-data-dict.dat", "rb") as fh:
            data_dict = pickle.load(fh)
    else:
        data_dict = generate(networks, args, args.dim)

    # grad_projs = data_dict["grad_projs"]
    grad_projs_unnorm = data_dict["grad_projs_unnorm"]

    with open("data/depth-data-dict.dat", "wb") as fh:
        pickle.dump(data_dict, fh)
    
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 0.8, args.max_depth))

    data = defaultdict(list)
    data_fig = defaultdict(dict)

    for (name, n_mistakes), grad_proj_series in grad_projs_unnorm.items():
        for idx in range(len(grad_proj_series)):
            if abs(grad_proj_series[idx]) > 10:
                grad_proj_series[idx] = grad_proj_series[idx - 1]
        # grad_projs_unnorm[name, n_mistakes] = medfilt(grad_proj_series)

        acc = 1 - (n_mistakes / args.dim)
        checker = [idx for idx, x in enumerate(grad_proj_series) if x > 0]
        crossing_point = checker[0] if len(checker) > 0 else None

        data[name].append((acc, crossing_point))
        data_fig[acc][name] = grad_proj_series

    for acc in data_fig:
        plt.figure()
        for (name, grad_proj_series), color in zip(data_fig[acc].items(), colors):
            plt.plot(grad_proj_series, label=name, color=color)
        plt.legend()
        plt.savefig(f"images/grad-proj-acc-{acc}.{args.format}")

    plt.figure()
    for color, name in zip(colors, data):
        plt.plot(*zip(*sorted(data[name])), label=name)

    plt.legend()
    plt.xlabel("Accuracy")
    plt.ylabel("Scale achieving proj equilibrium")
    plt.title("Scale achieving equilibrium by accuracy")
    plt.savefig(f"images/grad-project-by-depth.{args.format}")


def plot_grad_proj(
    args,
    scales,
    grad_projs,
    norms,
    n_mistakes: int,
    n_samples: int,
    legend: bool = False,
    unnorm: bool = True,
):
    acc = (n_samples - n_mistakes) / n_samples
    data = {
        name: projs
        for (name, n_mist), projs in grad_projs.items()
        if n_mistakes == n_mist
    }
    for name, projs in data.items():
        norm_seq = [norms[name] * scale for scale in scales[:-1]]
        plt.plot(norm_seq, projs[:-1], label=name)
    plt.xlabel("Scaled norm", fontsize=args.font_size)
    plt.ylabel("Gradient projection", fontsize=args.font_size)
    if unnorm:
        plt.title(f"Grad proj curve at {100 * acc:.0f}% acc", fontsize=args.font_size)
    else:
        plt.title(f"Grad proj curve at {100 * acc:.0f}% acc (normalized)", fontsize=args.font_size)
    if legend:
        plt.legend(fontsize=args.font_size)
    path = f"images/grad-proj/{100 * acc:.0f}.{args.format}"
    if not unnorm:
        path = path.replace(f".{args.format}", f"-norm.{args.format}")
    plt.savefig(path)
    print("Saved", path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--acc_min", type=int, default=.75)
    parser.add_argument("--acc_step", type=int, default=15)
    parser.add_argument("--font_size", type=str, default="large")
    parser.add_argument("--max_depth", type=int, default=13)
    parser.add_argument("--experiment", choices=["main", "depth"], default="main")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.experiment == "main":
        main(args)
    elif args.experiment == "depth":
        main_depth(args)
