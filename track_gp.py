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

from src.norm_sgd import NormSGD
from src.saturate import saturate
from src.loss import sequence_cross_entropy_with_logits
from src.si_transformer import SiEncoder, SiTransConfig
from src.language_model import transformers, LanguageModel
from src.tokenizer import Tokenizer
from src.utils import pad_sequence_to_len, get_mask


PATH = "/net/nfs.corp/allennlp/willm/data"

optims = {
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
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
    parser.add_argument("--dev_batch_size", type=int, default=50)
    parser.add_argument("--half", action="store_true")
    # Wikitext-2: Largest sentence is 699 on train, 429 on test.
    # Penn: Largest sentence is 82 on train, 74 on test.
    parser.add_argument("--seq_len", type=int, default=700)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--trans", type=str, default="vaswani", choices=["vaswani"] + list(transformers.keys())
    )
    parser.add_argument("--fig_dir", type=str, default="figs/finetune-trans")
    parser.add_argument("--data_dir", type=str, default="data/finetune-trans")
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--data", choices=["wikitext-2", "penn"], default="wikitext-2")

    parser.add_argument("--optim", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.)
    return parser.parse_args()


def train_model(
    args,
    model,
    train_tokens,
    train_mask,
    dev_tokens,
    dev_mask,
    optimizer,
    epochs=10,
    device="cuda:0",
):
    timeseries = defaultdict(list)
    best_loss = float("inf")

    for e in range(epochs):
        model.train()
        log.info(f"Starting epoch {e}...")
        perm = torch.randperm(len(train_tokens))
        train_tokens = train_tokens[perm, :]
        train_mask = train_mask[perm, :]

        params = [p.data.clone() for p in model.parameters() if p.requires_grad]

        for b in tqdm.trange(0, len(train_tokens) - args.batch_size, args.batch_size):
            prev_params = params

            batch_tokens = train_tokens[b : b + args.batch_size].to(device)
            batch_mask = train_mask[b : b + args.batch_size].to(device)
            optimizer.zero_grad()
            _, logits = model(batch_tokens[:, :-1])
            loss = sequence_cross_entropy_with_logits(
                logits, batch_tokens[:, 1:], batch_mask[:, :-1]
            )
            loss.backward()
            optimizer.step()

            params = [p.data.clone() for p in model.parameters() if p.requires_grad]
            deltas = [p - pp for p, pp in zip(params, prev_params)]

            pnorms = [p.norm(p=2).item() for p in params]
            dnorms = [d.norm(p=2).item() for d in deltas]
            projs = [(d.flatten() @ p.flatten()).item() for d, p in zip(params, deltas)]

            timeseries["pnorms"].append(pnorms)
            timeseries["dnorms"].append(dnorms)
            timeseries["projs"].append(projs)

        # model.eval()
        # metrics = get_metrics(args, model, dev_tokens, dev_mask, device=device)
        # for name, value in metrics.items():
        #     timeseries[name].append(value)
        # print(metrics)

        # Save the model checkpoint if this is the best performance yet.
        # if metrics["loss"] < best_loss:
        #     best_loss = metrics["loss"]
        #     data_dir = os.path.join(args.data_dir, args.data)
        #     ckpt_path = os.path.join(data_dir, args.trans + ".pt")
        #     torch.save(model.state_dict(), ckpt_path)

    return timeseries


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    model = model.to(device)

    if args.half:
        model = model.half()

    optim = optims[args.optim]

    timeseries = train_model(
        args,
        model,
        train_tokens,
        train_mask,
        dev_tokens,
        dev_mask,
        optim(model.parameters(), lr=args.lr, weight_decay=args.wd),
        epochs=args.epochs,
    )

    data_dir = "/net/nfs.corp/allennlp/willm/cached/wd"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = f"{data_dir}/{args.data}-{args.trans}-{args.optim}-lr={args.lr}-wd={args.wd}.dat"
    with open(filename, "wb") as fh:
        pickle.dump(timeseries, fh)

    print(f"Saved {filename}.")


if __name__ == "__main__":
    main(parse_args())
