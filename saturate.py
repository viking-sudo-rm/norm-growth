"""Analyze attention distribution in saturated saved transformer language models."""

from typing import List
from math import sqrt
import torch
from torch import nn
import argparse
from src.saturate import saturate
from src.si_transformer import SiTransConfig, SiEncoder, SiSelfAttention
import logging
from torch.nn.utils.rnn import pad_sequence
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from src.tokenizer import Tokenizer
from src.language_model import LanguageModel, transformers
from src.utils import pad_sequence_to_len, get_mask


PATH = "/net/nfs.corp/allennlp/willm/data"
MODELS = "/net/nfs.corp/allennlp/willm/models"
CACHED = "/net/nfs.corp/allennlp/willm/cached"

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
    parser.add_argument("--fine_lr", type=float, default=1e-1)
    parser.add_argument("--pre_epochs", type=int, default=5)
    parser.add_argument("--fine_epochs", type=int, default=0)
    parser.add_argument(
        "--trans", type=str, default="vaswani", choices=["vaswani"] + list(transformers.keys())
    )
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--data", choices=["wikitext-2", "penn"], default="wikitext-2")
    
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--n_samples", type=int, default=200)
    return parser.parse_args()


class AttnTracker:

    def __init__(self, cpu=True):
        self.attns = {}
        self.cpu = cpu

    def forward_hook(self, net, inputs, outputs):
        encodings = inputs[0]
        queries = net.query(encodings)
        keys = net.key(encodings)
        scores = torch.einsum("bti, bsi -> bts", queries, keys)

        if net.config.scale_scores:
            scores = scores / sqrt(net.d_head)

        if net.config.masked:
            seq_len = scores.size(1)
            arange = torch.arange(seq_len, device=queries.device)
            mask = arange.unsqueeze(dim=0) <= arange.unsqueeze(dim=1)
            scores = mask.unsqueeze(dim=0) * scores

        weights = net.softmax(scores)
        if self.cpu:
            weights = weights.cpu()
        self.attns[net.__name__] = weights


args = parse_args()

if not args.load:
    tokenizer = Tokenizer()

    log.info(f"Loading train data from {PATH}/{args.data}/train.txt...")
    raw_train = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/train.txt"))
    train_tokens = pad_sequence_to_len(raw_train, args.seq_len)
    train_mask = get_mask(raw_train, args.seq_len).float()

    log.info(f"Loading dev data from {PATH}/{args.data}/valid.txt...")
    raw_dev = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/valid.txt"))
    dev_tokens = pad_sequence_to_len(raw_dev, args.seq_len)
    dev_mask = get_mask(raw_dev, args.seq_len).float()

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

    model_path = f"{MODELS}/finetune-trans/{args.data}/{args.trans}.pt"
    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    model.to(dev)
    state_dict = torch.load(model_path, map_location=dev)
    model.load_state_dict(state_dict)

    print("Got the big boi vocab.", train_tokens.size())

    tracker = AttnTracker()
    for name, mod in model.named_modules():
        mod.__name__ = name
        if isinstance(mod, SiSelfAttention):
            mod.register_forward_hook(tracker.forward_hook)

    with saturate(model):
        tokens = dev_tokens[:args.n_samples, :].clone()
        tokens = tokens.to(dev)
        # Now the attention distribution should be recorded in the tracker.
        model(tokens)

    path = os.path.join(CACHED, "attn")
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/dev.dat", "wb") as fh:
        pickle.dump(tracker.attns, fh)
        print("Saved attention data.")
    attns = tracker.attns

else:
    print("Loading from pickle file...")
    path = os.path.join(CACHED, "attn")
    with open(f"{path}/dev.dat", "rb") as fh:
        attns = pickle.load(fh)

metrics = defaultdict(dict)
for name, tensor in attns.items():
    nonzero = (tensor > 1e-5).sum(dim=-1).float()
    metrics["std(n)"][name] = nonzero.std().item()
    metrics["split attention probability"][name] = (nonzero > 1).float().mean().item()
    metrics["# positions attended"][name] = nonzero.mean().item()

fig_dir = f"figs/attn/{args.data}/{args.trans}"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
log.info(f"Made {fig_dir}.")

title = "pre-norm" if args.trans == "pre_norm" else "post-norm"
for name, data in metrics.items():
    plt.figure()
    plt.hist(data.values(), bins=100)
    plt.title(f"{name} for all attention heads for {title}")
    plt.xlabel(name)
    plt.ylabel("# of attention heads")
    plt.savefig(f"{fig_dir}/{name}.pdf")
