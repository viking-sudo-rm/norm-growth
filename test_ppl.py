"""Compute perplexity of trainined models on the test set."""

import torch
from torch import nn
import argparse
from typing import List
import logging
from rich.logging import RichHandler

from src.language_model import LanguageModel, transformers
from src.tokenizer import Tokenizer
from src.utils import pad_sequence_to_len, get_mask
from src.loss import sequence_cross_entropy_with_logits
from src.saturate import saturate


DATA = os.getenv("DATA")
assert os.path.isdir(str(DATA)), f"Could not find data folder: {DATA}"
PATH = PATH
MODELS = os.getenv("MODELS")
assert os.path.isdir(str(MODELS)), f"Could not find models folder: {MODELS}"


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
    return parser.parse_args()


@torch.no_grad()
def get_norm(model):
    lins = [
        mod.weight
        for mod in model.modules()
        if isinstance(mod, nn.Linear) and mod.weight.requires_grad
    ]
    norms = torch.cat([lin.norm(p=2, dim=0) for lin in lins])
    return norms.mean()
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


def main(args):
    tokenizer = Tokenizer()

    log.info(f"Loading train data from {PATH}/{args.data}/train.txt...")
    raw_train = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/train.txt"))
    train_tokens = pad_sequence_to_len(raw_train, args.seq_len)
    train_mask = get_mask(raw_train, args.seq_len).float()

    log.info(f"Loading dev data from {PATH}/{args.data}/valid.txt...")
    raw_dev = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/valid.txt"))
    dev_tokens = pad_sequence_to_len(raw_dev, args.seq_len)
    dev_mask = get_mask(raw_dev, args.seq_len).float()

    log.info(f"Loading test data from {PATH}/{args.data}/test.txt...")
    raw_test = list(tokenizer.gen_tokens(f"{PATH}/{args.data}/test.txt", static=True))
    test_tokens = pad_sequence_to_len(raw_test, args.seq_len)
    test_mask = get_mask(raw_test, args.seq_len).float()

    max_test_len = max(len(s) for s in raw_test)
    assert max_test_len <= args.seq_len
    log.info(f"max test length is {max_test_len} (< {args.seq_len}).")

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

    model_path = f"data/finetune-trans/{args.data}/{args.trans}.pt"
    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    model.to(dev)
    state_dict = torch.load(model_path, map_location=dev)
    model.load_state_dict(state_dict)

    metrics = get_metrics(args, model, test_tokens, test_mask, device=dev)
    log.info(f"pplx={metrics['pplx']:.2f}, sat={metrics['sat']:.2f}")


if __name__ == "__main__":
    main(parse_args())
