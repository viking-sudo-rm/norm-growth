import tqdm
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from itertools import product
import argparse
from copy import deepcopy
import numpy as np

from src.norm_sgd import NormSGD
from src.saturate import saturate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_bits", type=int, default=10)
    parser.add_argument("--hid_max", type=int, default=15)
    parser.add_argument("--hid_step", type=int, default=1)
    return parser.parse_args()


class AutoEncoder(nn.Module):
    def __init__(self, in_bytes, out_bytes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(1 << in_bytes, out_bytes),
            nn.Tanh(),
        )
        self.decoder = nn.Linear(out_bytes, 1 << in_bytes, bias=False)

    def encode(self, x: torch.Tensor) -> torch.FloatTensor:
        return self.encoder(x)

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        return self.decoder(z)

    def forward(self, x):
        encoding = self.encode(x)
        return encoding, self.decode(encoding)


def generate_data(num_bits: int, device=None):
    return torch.arange(0, 1 << num_bits, device=device)


def get_norm(model):
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    return torch.cat([p.flatten() for p in params]).norm(p=2)


def get_saturation(model, data):
    soft = model.encode(data)
    with saturate(model):
        hard = model.encode(data)

    prod = torch.einsum("bi, bi -> b", soft, hard)
    soft_norm = soft.norm(p=2, dim=-1)
    hard_norm = hard.norm(p=2, dim=-1)
    sats = prod / (soft_norm * hard_norm + 1e-9)
    return sats.mean()


def train_model(args, model, data, optimizer, criterion, epochs=10):
    model.train()

    timeseries = defaultdict(list)
    for e in tqdm.tqdm(range(epochs), ascii=True):
        perm = torch.randperm(len(data))
        data = data[perm]

        for b in range(0, len(data), args.batch_size):
            batch = data[b : b + args.batch_size]
            optimizer.zero_grad()
            encoding, output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

        metrics = {
            "acc1": (output.argmax(dim=-1) == data).float().mean().item(),
            "norm": get_norm(model).item(),
            "loss": loss.item(),
            "sat": get_saturation(model, data).item(),
        }
        for name, value in metrics.items():
            timeseries[name].append(value)
        # tqdm.tqdm.write(f"e{e}:", ", ".join(f"{n}={v}" for n, v in metrics.items()))

    return timeseries


def main(args):
    data = generate_data(args.n_bits, device=args.gpu)
    all_series = defaultdict(dict)

    for hid_bits in range(1, args.hid_max, args.hid_step):
        model = AutoEncoder(args.n_bits, hid_bits)
        model.cuda(args.gpu)
        criterion = nn.CrossEntropyLoss()

        all_series[hid_bits]["pre"] = train_model(
            args, model, data, optim.AdamW(model.parameters()), criterion, epochs=100
        )
        all_series[hid_bits]["fine"] = train_model(
            args,
            model,
            data,
            NormSGD(model.parameters(), lr=1e-3),
            criterion,
            epochs=100,
        )

        all_series[hid_bits]["all"] = deepcopy(all_series[hid_bits]["pre"])
        for key, series in all_series[hid_bits]["fine"].items():
            all_series[hid_bits]["all"][key].extend(series)


    colors = plt.get_cmap("hsv")(np.linspace(.2, .8, 15))  #"Greys"
    for metric in all_series[1]["all"].keys():
        plt.figure()
        minval, maxval = None, None
        for color, (hid_bits, data) in zip(colors, all_series.items()):
            values = data["all"][metric]
            plt.plot(values, label=str(hid_bits), color=color)
            minval = min(minval, min(values)) if minval is not None else min(values)
            maxval = max(maxval, max(values)) if maxval is not None else max(values)
        split = len(data["pre"][metric]) - 1
        plt.plot([split, split], [minval, maxval], "k-", lw=2)
        plt.legend()
        plt.title(f"{metric} over time")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.hsv()
        plt.savefig(f"figs/finetune/{metric}.pdf")


if __name__ == "__main__":
    main(parse_args())
