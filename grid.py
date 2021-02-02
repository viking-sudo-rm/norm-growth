"""Do grid search for norm growth."""

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
import time
from multiprocessing import Process, Queue

from src.norm_sgd import NormSGD
from src.saturate import saturate
from src.loss import sequence_cross_entropy_with_logits
from src.si_transformer import SiEncoder, SiTransConfig
from src.language_model import transformers, LanguageModel
from src.tokenizer import Tokenizer
from src.utils import pad_sequence_to_len, get_mask


PATH = "/net/nfs.corp/allennlp/willm/data"
SAVE = "/net/nfs.corp/allennlp/willm/models/grid"

optims = {
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
}

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)

    # Wikitext-2: Largest sentence is 699 on train, 429 on test.
    # Penn: Largest sentence is 82 on train, 74 on test.
    parser.add_argument("--seq_len", type=int, default=700)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--trans", type=str, default="vaswani", choices=["vaswani"] + list(transformers.keys())
    )
    parser.add_argument("--fig_dir", type=str, default="figs/finetune-trans")
    parser.add_argument("--data_dir", type=str, default="grid-norms")
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--data", choices=["wikitext-2", "penn"], default="wikitext-2")
    parser.add_argument("--wd_range", type=int, default=5)
    parser.add_argument("--lr_range", type=int, default=5)
    parser.add_argument("--optim", default="SGD", choices=optims.keys())
    parser.add_argument("--n_gpus", type=int)
    return parser.parse_args()


@torch.no_grad()
def get_norm_linear(model):
    lins = [
        mod.weight
        for mod in model.modules()
        if isinstance(mod, nn.Linear) and mod.weight.requires_grad
    ]
    norms = torch.cat([lin.norm(p=2, dim=0) for lin in lins])
    return norms.mean()


@torch.no_grad()
def get_norm_encoder(model):
    params = model.encoder.parameters()
    params = torch.cat([p.flatten() for p in params])
    return params.norm(p=2).item()


def train_model(
    args,
    model,
    train_tokens,
    train_mask,
    optimizer,
    epochs=5,
    device="cuda:0",
):
    for e in range(epochs):
        model.train()
        log.info(f"Starting epoch {e}...")
        perm = torch.randperm(len(train_tokens))
        train_tokens = train_tokens[perm, :]
        train_mask = train_mask[perm, :]

        for b in tqdm.trange(0, len(train_tokens) - args.batch_size, args.batch_size):
            batch_tokens = train_tokens[b : b + args.batch_size].to(device)
            batch_mask = train_mask[b : b + args.batch_size].to(device)
            optimizer.zero_grad()
            _, logits = model(batch_tokens[:, :-1])
            loss = sequence_cross_entropy_with_logits(
                logits, batch_tokens[:, 1:], batch_mask[:, :-1]
            )
            loss.backward()
            optimizer.step()


def wrapper(optim, lr, wd, device, tokenizer, train_tokens, train_mask):
    log.info("Creating model...")
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

    log.info(f"Moving model to {device}.")
    model = model.to(device)

    norms = defaultdict(list)
    norms["linear"].append(get_norm_linear(model))
    norms["encoder"].append(get_norm_encoder(model))

    log.info(f"Start training on {device}...")
    train_model(
        args,
        model,
        train_tokens,
        train_mask,
        optim(model.parameters(), lr=lr, weight_decay=wd),
        epochs=args.epochs,
        device=device,
    )

    norms["linear"].append(get_norm_linear(model))
    norms["encoder"].append(get_norm_encoder(model))
    return norms


def run_exp(gpu_num, in_queue):
    # It seems like thread locking or something, gross.
    tokenizer = Tokenizer()
    log.info(f"Loading train data from {PATH}/{args.data}/train.txt...")
    raw_train = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/train.txt"))
    train_tokens = pad_sequence_to_len(raw_train, args.seq_len)
    train_mask = get_mask(raw_train, args.seq_len).float()
    train_len = max(len(s) for s in raw_train)
    assert train_len <= args.seq_len
    log.info(f"Max train sentence length is {train_len} (<= {args.seq_len}).")

    while not in_queue.empty():
        try:
            experiment = in_queue.get(timeout=3)
        except:
            return
        before = time.time()

        optim = experiment["optim"]
        lr = experiment["lr"]
        wd = experiment["wd"]
        device = f"cuda:{gpu_num}"

        norms = wrapper(optim, lr, wd, device, tokenizer, train_tokens, train_mask)
        name = f"{optim.__name__}-lr={lr}-wd={wd}.dat"
        path = os.path.join(SAVE, name)
        if not os.path.isdir(SAVE):
            os.makedirs(SAVE)
        with open(path, "wb") as fh:
            pickle.dump(norms, fh)

        with open("output.txt", "a+") as f:
            f.write(
                "Finished experiment {} in {}.\n".format(
                    experiment, str((time.time() - before) / 60.0)
                )
            )


def main():
    gpus = [str(x) for x in range(args.n_gpus)]
    gpus.extend(gpus)

    optim = optims[args.optim]
    # optims = [torch.optim.SGD, torch.optim.AdamW]
    lrs = np.logspace(1, args.lr_range, base=.1, num=20)
    wds = np.logspace(1, args.wd_range, base=.1, num=20)

    experiments = []
    for optim, lr, wd in product([optim], lrs, wds):
        experiments.append({"optim": optim, "lr": lr, "wd": wd})
    print(experiments)

    queue = Queue()
    for e in experiments:
        queue.put(e)
    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(gpu, queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    global args
    args = parse_args()
    main()
