"""Script for training transformers with potential architectural modifications."""

from typing import List
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
import logging
from rich.logging import RichHandler
from rich import print
import pickle
import os
from torch.autograd import grad

from transformers.optimization import Adafactor

from schedulers import get_policy
from src.norm_sgd import NormSGD
from src.saturate import saturate
from src.loss import sequence_cross_entropy_with_logits
from src.language_model import transformers, LanguageModel
from src.tokenizer import Tokenizer
from src.utils import pad_sequence_to_len, get_mask


DATA = os.getenv("DATA")
assert os.path.isdir(str(DATA)), f"Could not find data folder: {DATA}"
PATH = DATA
MODELS = os.getenv("MODELS")
assert os.path.isdir(str(MODELS)), f"Could not find models folder: {MODELS}"


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


optims = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adafactor": Adafactor,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dev_batch_size", type=int, default=50)
    parser.add_argument("--half", action="store_true")
    # Wikitext-2: Largest sentence is 699 on train, 429 on test.
    # Penn: Largest sentence is 82 on train, 74 on test.
    parser.add_argument("--seq_len", type=int, default=700)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--fine_lr", type=float, default=1e-1)
    parser.add_argument("--pre_epochs", type=int, default=5)
    parser.add_argument("--fine_epochs", type=int, default=0)
    parser.add_argument(
        "--trans", type=str, default="vaswani", choices=["vaswani"] + list(transformers.keys())
    )
    parser.add_argument("--fig_dir", type=str, default="figs/finetune-trans")
    parser.add_argument("--data_dir", type=str, default=f"{MODELS}/finetune-trans")
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--data", choices=["wikitext-2", "penn"], default="wikitext-2")
    parser.add_argument("--optim", choices=optims.keys(), default="adamw")
    parser.add_argument("--sched", choices=["constant_lr", "linear_lr", "sqrt_lr"], default="constant_lr")
    parser.add_argument("--stop_iteration", type=int, default=1000)  # End of constant LR warmup
    parser.add_argument("--batch_metrics", type=int, default=None)

    return parser.parse_args()


@torch.no_grad()
def get_norm(model):
    # Use the same norm as for T5.
    params = [p for p in model.parameters() if p.requires_grad]
    params = [p for p in params if len(p.shape) > 0]
    return torch.cat([p.flatten() for p in params]).norm(p=2)

    # lins = [
    #     mod.weight
    #     for mod in model.modules()
    #     if isinstance(mod, nn.Linear) and mod.weight.requires_grad
    # ]
    # norms = torch.cat([lin.norm(p=2, dim=0) for lin in lins])
    # return norms.mean()

    # params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    # return torch.cat([p.flatten() for p in params]).norm(p=2)


@torch.no_grad()
def get_saturation(soft, model, hard_callback):
    with saturate(model):
        hard = hard_callback()
    prod = torch.einsum("bti, bti -> bt", soft, hard)
    soft_norm = soft.norm(p=2, dim=-1)
    hard_norm = hard.norm(p=2, dim=-1)
    return prod / (soft_norm * hard_norm + 1e-9)


def get_projection(loss, params):
    grads = grad(loss.sum(), params)
    params = torch.cat([p.flatten() for p in params])
    grads = torch.cat([g.flatten() for g in grads])
    proj = params.T @ grads
    proj_sim = proj / (params.norm(p=2) * grads.norm(p=2) + 1e-9)
    return proj, proj_sim


@torch.no_grad()
def get_metrics(args, model, dev_tokens, dev_mask, device="cuda:0"):
    all_agreement = []
    all_saturation = []
    all_loss = []
    all_proj = 0.
    all_proj_sim = 0.
    # In this loop, we iterate over the full dev set, including the small bit at the end.
    for b in range(0, len(dev_tokens), args.dev_batch_size):
        dev_batch_tokens = dev_tokens[b : b + args.dev_batch_size].to(device)
        dev_batch_mask = dev_mask[b : b + args.dev_batch_size].to(device)
        dev_encoding, dev_logits = model(dev_batch_tokens[:, :-1])
        dev_loss = sequence_cross_entropy_with_logits(
            dev_logits, dev_batch_tokens[:, 1:], dev_batch_mask[:, :-1], average=None
        )
        dev_preds = dev_logits.argmax(dim=-1)
        agreement = (dev_preds == dev_batch_tokens[:, 1:]).float() * dev_batch_mask[
            :, :-1
        ]
        saturation = get_saturation(
            dev_encoding * dev_batch_mask[:, :-1].unsqueeze(dim=-1),
            model,
            lambda: model(dev_batch_tokens[:, :-1])[0]
            * dev_batch_mask[:, :-1].unsqueeze(dim=-1),
        )
        # With multiple devices, this doesn't work. Also can turn on no_grad when we don't do this.
        # proj, proj_sim = get_projection(dev_loss, [p for p in model.parameters() if p.requires_grad])
        
        all_loss.append(dev_loss.cpu())
        all_agreement.append(agreement.cpu())
        all_saturation.append(saturation.cpu())
        # all_proj += proj.item()
        # all_proj_sim += proj_sim.item()

    all_loss = torch.cat(all_loss, dim=0)
    all_agreement = torch.cat(all_agreement, dim=0)
    all_saturation = torch.cat(all_saturation, dim=0)
    all_perps = torch.pow(2, all_loss)
    numel = dev_mask[:, :-1].sum()

    return {
        "acc1": (all_agreement.sum() / numel).item(),
        "norm": get_norm(model).item(),
        "loss": all_loss.mean().item(),
        "pplx": all_perps.mean().item(),
        "sat": (all_saturation.sum() / numel).item(),
        # "proj": all_proj / len(dev_tokens),
        # "proj_sim": all_proj_sim / len(dev_tokens),
    }


def train_model(
    args,
    model,
    train_tokens,
    train_mask,
    dev_tokens,
    dev_mask,
    optimizer,
    epochs=10,
    record_init=False,
    device="cuda:0",
    scheduler: str = None,
    max_iterations = None,
):
    batch_timeseries = defaultdict(list)
    timeseries = defaultdict(list)
    if record_init:
        metrics = get_metrics(args, model, dev_tokens, dev_mask)
        for name, value in metrics.items():
            timeseries[name].append(value)
        print(metrics)

    best_loss = float("inf")

    lr_adjuster = get_policy(scheduler)(optimizer, args, max_iterations=max_iterations)

    iteration = 0
    for e in range(epochs):

        model.train()
        log.info(f"Starting epoch {e}...")
        perm = torch.randperm(len(train_tokens))
        train_tokens = train_tokens[perm, :]
        train_mask = train_mask[perm, :]

        for b in tqdm.trange(0, len(train_tokens) - args.batch_size, args.batch_size):
            cur_lr = lr_adjuster(e, iteration)

            if args.batch_metrics is not None and iteration % args.batch_metrics == 0:
                norm = get_norm(model).item()
                batch_timeseries["step"].append(iteration)
                batch_timeseries["norm"].append(norm)
                batch_timeseries["lr"].append(cur_lr)

            tqdm.tqdm.write(f"i={iteration}, lr={cur_lr}", end="\r")
            batch_tokens = train_tokens[b : b + args.batch_size].to(device)
            batch_mask = train_mask[b : b + args.batch_size].to(device)
            optimizer.zero_grad()
            _, logits = model(batch_tokens[:, :-1])
            loss = sequence_cross_entropy_with_logits(
                logits, batch_tokens[:, 1:], batch_mask[:, :-1]
            )
            loss.backward()
            optimizer.step()
            iteration += 1

        model.eval()
        metrics = get_metrics(args, model, dev_tokens, dev_mask, device=device)
        for name, value in metrics.items():
            timeseries[name].append(value)
        print(metrics)

        # Save the model checkpoint if this is the best performance yet.
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            data_dir = os.path.join(args.data_dir, args.data)
            ckpt_path = os.path.join(data_dir, args.trans + ".pt")
            torch.save(model.state_dict(), ckpt_path)

    return timeseries, batch_timeseries


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_series = {}
    tokenizer = Tokenizer()

    # TODO: Added this in utils.
    log.info(f"Loading train data from {PATH}/{args.data}/train.txt...")
    raw_train = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/train.txt"))
    train_tokens = pad_sequence_to_len(raw_train, args.seq_len)
    train_mask = get_mask(raw_train, args.seq_len).float()
    train_len = max(len(s) for s in raw_train)
    assert train_len <= args.seq_len
    log.info(f"Max train sentence length is {train_len} (<= {args.seq_len}).")

    log.info(f"Loading dev data from {PATH}/{args.data}/valid.txt...")
    raw_dev = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/valid.txt"))
    dev_tokens = pad_sequence_to_len(raw_dev, args.seq_len)
    dev_mask = get_mask(raw_dev, args.seq_len).float()
    dev_len = max(len(s) for s in raw_dev)
    assert dev_len <= args.seq_len
    log.info(f"Max dev sentence length is {dev_len} (<= {args.seq_len}).")

    # Maximum number of training steps, used for linearly decaying learning rate schedule.
    max_iterations = len(raw_train) // args.batch_size * args.pre_epochs

    model = LanguageModel(
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_vocab=tokenizer.d_vocab,
        seq_len=args.seq_len,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        encoder_type=args.trans,
        bias=not args.no_bias,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        # model = nn.DataParallel(model, device_ids=list(range(args.gpus)))
    model = model.to(device)

    if args.half:
        model = model.half()
    
    opt = optims[args.optim]

    all_series["pre"], batch_data = train_model(
        args,
        model,
        train_tokens,
        train_mask,
        dev_tokens,
        dev_mask,
        opt(model.parameters(), lr=args.lr),
        epochs=args.pre_epochs,
        record_init=True,
        scheduler=args.sched,
        max_iterations=max_iterations,
    )
    # TODO: Should we try varying the learning rate from this checkpoint?
    all_series["fine"], batch_data_fine = train_model(
        args,
        model,
        train_tokens,
        train_mask,
        dev_tokens,
        dev_mask,
        NormSGD(model.parameters(), lr=args.fine_lr),
        epochs=args.fine_epochs,
    )

    all_series["all"] = deepcopy(all_series["pre"])
    for metric, series in all_series["fine"].items():
        all_series["all"][metric].extend(series)
    
    data_dir = os.path.join(args.data_dir, args.data)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    filename = f"{args.trans}-{args.optim}-{args.sched}"

    data_path = os.path.join(data_dir, filename + ".dat")
    with open(data_path, "wb") as fh:
        pickle.dump(all_series, fh)
    
    data_path = os.path.join(data_dir, f"{filename}-batch_data.dat")
    with open(data_path, "wb") as fh:
        pickle.dump(batch_data, fh)
    
    ckpt_path = os.path.join(data_dir, f"{filename}.pt")
    torch.save(model.state_dict(), ckpt_path)

    fig_dir = os.path.join(args.fig_dir, args.data, filename)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    for metric, values in all_series["all"].items():
        plt.figure()
        minval, maxval = None, None
        plt.plot(values)
        if len(all_series["fine"][metric]) > 0:
            min_val = min(values)
            max_val = max(values)
            split = len(all_series["pre"][metric]) - 1
            plt.plot([split, split], [min_val, max_val], "k-", lw=2)
        plt.title(f"dev {metric} over training")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.savefig(f"{fig_dir}/{metric}.pdf")


if __name__ == "__main__":
    main(parse_args())
