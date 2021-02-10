import argparse
import os
import random
import torch
import numpy as np
from collections import defaultdict
import tqdm
from rich import print
import matplotlib.pyplot as plt

from src.saturate import saturate
from src.language_model import transformers, LanguageModel


DATA = os.getenv("DATA")
assert os.path.isdir(str(DATA)), f"Could not find data folder: {DATA}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--fig_dir", type=str, default="figs/trans-scale")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--n_sents", type=int, default=10)
    parser.add_argument("--d_vocab", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    sents = torch.randint(high=args.d_vocab, size=[args.n_sents, args.seq_len])

    encoders = ["pre_norm", "control"]
    biases = [True, False]
    scales = np.linspace(1, 10, 20)
    results = defaultdict(list)
    prog = tqdm.tqdm(total=len(encoders) * len(biases) * (len(scales) + 1))
    for encoder in encoders:
        for bias in biases:
            model = LanguageModel(
                d_model=args.d_model,
                d_ff=args.d_ff,
                d_vocab=args.d_vocab,
                seq_len=args.seq_len,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                encoder_type=encoder,
                bias=bias,
            )
            with saturate(model):
                sat_encodings, _ = model(sents)
                sat_encodings = sat_encodings.flatten()
                prog.update()
            
            for scale in scales:
                with saturate(model, infinity=scale):
                    encodings, _ = model(sents)
                    encodings = encodings.flatten()
                sim = encodings @ sat_encodings / (encodings.norm(p=2) * sat_encodings.norm(p=2))
                results[encoder, bias].append(sim.item())
                prog.update()
    prog.close()

    plt.rcParams.update({
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "legend.fontsize": "large",
    })
    plt.plot(scales, results["pre_norm", False], color="orange", label="Pre-norm")
    plt.plot(scales, results["pre_norm", True], color="orange", linestyle="dashed", label="Pre-norm (+bias)")
    plt.plot(scales, results["control", False], color="blue", label="Post-norm")
    plt.plot(scales, results["control", True], color="blue", linestyle="dashed", label="Post-norm (+bias)")
    plt.legend()
    plt.title("Scaling curves for transformer variants")
    plt.xlabel(R"Scaling factor $c$")
    plt.ylabel(R"Cosine similarity to sat. transformer")

    if not os.path.isdir(args.fig_dir):
        os.makedirs(args.fig_dir)
    path = f"{args.fig_dir}/scales.pdf"
    plt.savefig(path)
    print(f"[green]=>[/green] Saved {path}.")


if __name__ == "__main__":
    main(parse_args())
