# import seaborn as sns
from typing import Tuple, List
from argparse import ArgumentParser
import torch
from torch.nn import Parameter
from transformers import *
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import random
from rich import print
from torch.nn.utils.rnn import pad_sequence
import warnings
import pickle

from src.utils.saturate import saturate
from src.metrics.param_norm import ParamNorm
from src.utils.huggingface import (
    cos,
    get_activation_norms,
    get_weight_norms,
    get_paired_mag_and_act_norms,
    get_params_by_layer,
    get_prunable_parameters,
    get_tokenizer_and_model,
    wrap_contextualize,
)

# See https://github.com/huggingface/transformers/issues/37.
PATH = "images/sim-by-layer"


class Avg:
    def __init__(self):
        self.sum = 0
        self.num = 0
    
    def update(self, tensor: torch.Tensor):
        self.sum += tensor.sum().item()
        self.num += tensor.numel()
    
    def get(self):
        return self.sum / self.num


class WrapT5(torch.nn.Module):
    def __init__(self, model_name: str, random_init: bool = False):
        super().__init__()
        if random_init:
            config = T5Config.from_pretrained(model_name, output_hidden_states=True)
            self.t5 = T5Model(config)
        else:
            self.t5 = T5Model.from_pretrained(model_name, output_hidden_states=True)

    def forward(self, input_ids):
        results = self.t5(input_ids, decoder_input_ids=input_ids)
        return results[:3]


class WrapXLNet(torch.nn.Module):
    def __init__(self, model_name: str, random_init: bool = False):
        super().__init__()
        if random_init:
            config = XLNetConfig.from_pretrained(model_name, output_hidden_states=True)
            self.xlnet = XLNetModel(config)
        else:
            self.xlnet = XLNetModel.from_pretrained(model_name, output_hidden_states=True)

    def forward(self, input_ids):
        results = self.xlnet(input_ids)
        return results[0], results[0], results[1]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", action="append")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mock_sents", action="store_true")
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--num_sents", type=int, default=100)
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    return parser.parse_args()


def get_sentences(args):
    if args.mock_sents:
        return [
            "Hello to my little friend.",
            "It's a great day in Seattle, besides the virus.",
            "Working from home is great.",
            "Wow, who needs pre-annotated corpora?",
        ]

    with open("/home/willm/data/brown.txt") as fh:
        text = fh.read()
        sentences = [line for line in text.split("\n\n") if not line.startswith("#")]
        return sentences[:args.num_sents]

def collect_data(args):
    sentences = get_sentences(args)

    tokenizers = [
        AutoTokenizer.from_pretrained("bert-base-cased"),
        AutoTokenizer.from_pretrained("roberta-base"),
        AutoTokenizer.from_pretrained("t5-base"),
        AutoTokenizer.from_pretrained("xlnet-base-cased"),
    ]

    model_names = ["bert-base-cased", "roberta-base", "t5-base", "xlnet-base-cased"]

    if not args.random_init:
        models = [
            BertModel.from_pretrained("bert-base-cased", output_hidden_states=True),
            RobertaModel.from_pretrained("roberta-base", output_hidden_states=True),
            WrapT5("t5-base"),
            WrapXLNet("xlnet-base-cased"),
        ]
    
    else:
        models = [
            BertModel(BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)),
            RobertaModel(RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)),
            WrapT5("t5-base", random_init=True),
            WrapXLNet("xlnet-base-cased", random_init=True),
        ]

    # tokenizers = [
    #     # BertTokenizer.from_pretrained("bert-base-cased"),
    #     AutoModel.from_pretrained("roberta-base"),
    #     # XLNetTokenizer.from_pretrained("xlnet-base-cased"),
    # ]

    # models = [
    #     # BertForMaskedLM.from_pretrained("bert-base", output_hidden_states=True),
    #     AutoModel.from_pretrained("roberta-base", output_hidden_states=True),
    #     # RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True),
    #     # XLNetLMHeadModel.from_pretrained("xlnet-base-cased"),
    # ]

    sims_by_model = {}

    for name, tokenizer, model in zip(model_names, tokenizers, models):
        print(f"[green]=>[/green] {type(model).__name__}...")

        sim_avgs = [Avg() for _ in range(13)]

        for sentence in sentences:
            input_ids = torch.tensor(tokenizer.encode(sentence, max_length=512)).unsqueeze(dim=0)

        # results = tokenizer.batch_encode_plus(
        #     sentences, max_length=512, pad_to_max_length=True, return_tensors="pt"
        # )
        # input_ids = results["input_ids"]

            pool, final, states = model(input_ids)
            with saturate(model):
                hard_pool, hard_final, hard_states = model(input_ids)

            assert isinstance(states, tuple)
            assert isinstance(hard_states, tuple)
            sims = [
                cos(state, hard_state)
                for state, hard_state in zip(states, hard_states)
            ]

            for sim, avg in zip(sims, sim_avgs):
                avg.update(sim)

        for layer, avg in enumerate(sim_avgs):
            print(f"[red]Layer #{layer} Sim[/red]: {avg.get():.2f}")

        sims_by_model[name] = [avg.get() for avg in sim_avgs]
    
    return sims_by_model


def main(args):
    if args.load is not None:
        with open(args.load, "rb") as fh:
            sims_by_model = pickle.load(fh)
    else:
        sims_by_model = collect_data(args)
    
    layers = list(range(1, 13))

    import matplotlib.pyplot as plt
    for model, data in sims_by_model.items():
        data = data[1:]  # Ignore the embedding layer, which is constant 1.
        plt.plot(layers, data, label=model)
    plt.ylim(0, 1)
    plt.xlabel("Layer #")
    plt.ylabel("Representation similarity")
    plt.title("Randomly initialized representation similarity" if args.random_init else "Pretrained representation similarity")
    plt.legend()
    if args.random_init:
        path = os.path.join(PATH, "random-init.png")
    else:
        path = os.path.join(PATH, "pretrained.png")
    plt.savefig(path)
    print(f"[green]=>[/green] Saved fig to {path}.")

    if not args.load:
        with open(f"data/sims_by_model-{args.random_init}.dat", "wb") as fh:
            pickle.dump(sims_by_model, fh)


if __name__ == "__main__":
    main(parse_args())
