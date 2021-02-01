"""Access pickled training metrics for trained transformer models."""

import pickle
import os
from rich import print
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/finetune-trans/wikitext-2")
    return parser.parse_args()


args = parse_args()

for name in os.listdir(args.data_dir):
    if not name.endswith(".dat"):
        continue
    trans = name.replace(".dat", "")
    with open(f"{args.data_dir}/{name}", "rb") as fh:
        data = pickle.load(fh)
        idx, min_ppl = min(enumerate(data["pre"]["pplx"]), key=lambda tup: tup[1])
        sat = data["pre"]["sat"][idx]
        print(f"{name}: ppl={min_ppl:.2f}, sat={sat:.2f}")

# Notes: constantly check for `lookahead` that breaks the transformer.
# Why does the control get to 100% saturation? Is something off with the architecture?
# https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word
# SOTA without external data is a BERT-like model, getting ~31ppl.