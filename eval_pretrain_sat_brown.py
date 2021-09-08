from argparse import ArgumentParser
import torch
from transformers import *
import matplotlib.pyplot as plt
import os
from rich import print
import pickle

from src.saturate import saturate


# See https://github.com/huggingface/transformers/issues/37.
PATH = "figs/sim-by-layer"
DATA = os.getenv("DATA")
CACHED = os.getenv("CACHED")


def cos(vec1: torch.FloatTensor, vec2: torch.FloatTensor, dim: int = -1) -> torch.FloatTensor:
    """Return the cosine similarity between two vectors along `dim`.
    
    There was a bug in this that I fixed."""
    return torch.sum(vec1 * vec2, dim=dim) / (
        vec1.norm(dim=dim) * vec2.norm(dim=dim) + 1e-9
    )


class Avg:
    def __init__(self):
        self.sum = 0.
        self.num = 0
    
    def update(self, tensor: torch.Tensor):
        self.sum += tensor.sum().item()
        self.num += tensor.numel()
    
    @property
    def value(self):
        return self.sum / self.num
    
    @property
    def is_zero(self):
        return self.sum == 0. and self.num == 0


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

    with open(f"{DATA}/brown.txt") as fh:
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

    sims_by_model = {}
    for name, tokenizer, model in zip(model_names, tokenizers, models):
        print(f"[green]=>[/green] {type(model).__name__}...")
        sim_avgs = [Avg() for _ in range(13)]

        for sentence in sentences:
            input_ids = torch.tensor(tokenizer.encode(sentence, max_length=512, truncation=True)).unsqueeze(dim=0)
            outputs = model(input_ids)
            _, _, states = outputs if isinstance(outputs, tuple) else outputs.values()
            with saturate(model):
                hard_outputs = model(input_ids)
                _, _, hard_states = hard_outputs if isinstance(hard_outputs, tuple) else hard_outputs.values()
            for state, hard_state, avg in zip(states, hard_states, sim_avgs):
                sim = cos(state, hard_state)
                avg.update(sim)

        sims_by_model[name] = [avg.value for avg in sim_avgs if not avg.is_zero]
        for layer, avg_value in enumerate(sims_by_model[name]):
            print(f"[red]Layer #{layer} Sim[/red]: {avg_value:.2f}")

    return sims_by_model


def main(args):
    if args.load is not None:
        with open(args.load, "rb") as fh:
            sims_by_model = pickle.load(fh)
    else:
        sims_by_model = collect_data(args)

    for model, data in sims_by_model.items():
        data = data[1:]  # Ignore the embedding layer, which is constant 1.
        layers = list(range(1, len(data) + 1))
        plt.plot(layers, data, label=model, marker="o")
    plt.ylim(top=1)
    plt.xlabel("Layer #")
    plt.ylabel("Representation similarity")
    plt.title("Randomly initialized representation similarity" if args.random_init else "Pretrained representation similarity")
    plt.legend()
    plt.tight_layout()
    if args.random_init:
        path = os.path.join(PATH, "random-init.pdf")
    else:
        path = os.path.join(PATH, "pretrained.pdf")
    plt.savefig(path)
    print(f"[green]=>[/green] Saved fig to {path}.")

    if not args.load:
        with open(f"{CACHED}/sims_by_model-{args.random_init}.dat", "wb") as fh:
            pickle.dump(sims_by_model, fh)


if __name__ == "__main__":
    main(parse_args())
