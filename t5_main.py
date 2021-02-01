"""Script to analyze weight trajectory during training of T5.

To download the model checkpoints from GCP, you can do:
gsutil -m cp -r gs://t5-data/pretrained_models/ data/t5-models

But also, this should work if you set INIT_CHECKPONT to a GCP path.
"""

# NLP IS SO WEIRD EVERYONE KNOWS T5 IS A TERMINATOR -> https://terminator.fandom.com/wiki/T-5000

from typing import Iterable
import os
import numpy as np
from math import sqrt
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pickle
import tqdm
from collections import defaultdict
import argparse

from t5.models.mtf_model import MtfModel
import t5.data
import gin


# MIXTURE_NAME = 'all_mix'
MIXTURE_NAME = "c4_v020_unsupervised"

CKPT_PATH = "/home/willm/data/bsl/bsl-0/checkpoint"
FILE_FORMAT = """model_checkpoint_path: "{ckpt}"
all_model_checkpoint_paths: "{ckpt}"
"""

FILTER_PRED = lambda name: name.startswith("shared/")


def get_checkpoints(model_dir):
    # return tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
    ckpts = []
    for file_name in os.listdir(model_dir):
        if file_name.endswith(".index"):
            ckpts.append(file_name.replace(".index", ""))
    ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[1]))
    return list(ckpts)


def downsample(li: list, samples: int = 5):
    step = len(li) // samples
    for idx, item in enumerate(li):
        if idx % step == 0:
            yield item


def write_checkpoint_file(ckpt):
    with open(CKPT_PATH, "w") as fh:
        contents = FILE_FORMAT.format(ckpt=ckpt)
        fh.write(contents)


def _operative_config_path(model_dir):
    return os.path.join(model_dir, "operative_config.gin")


def _fan_in(shape) -> int:
    # This is from some TensorFlow code or something.
    return float(shape[-2]) if len(shape) > 1 else float(shape[-1])


def get_param_names(estimator):
    return [
        p
        for p in estimator.get_variable_names()
        if not FILTER_PRED(p)
    ]


def filter_by_layer(param_names, layer_num: int):
    expr = f"encoder/block_{layer_num:03d}"
    return [p for p in param_names if p.startswith(expr)]


def get_param_norm(params: Iterable[np.ndarray], normalize: bool = False):
    # There are weird scalars in here, which we filter out.
    values = [v for v in params if len(v.shape) > 0]
    if normalize:
        values = [value / sqrt(_fan_in(value.shape)) for value in values]
    flat = np.concatenate([value.flatten() for value in values])
    norm = np.linalg.norm(flat)
    return norm


def get_param(params: Iterable[np.ndarray]):
    values = [v for v in params if len(v.shape) > 0]
    return np.concatenate([value.flatten() for value in values])


def get_all_norms_normalized_WRONG(estimator):
    """FIXME: No sqrt boi."""
    params = [
        estimator.get_variable_value(p)
        for p in estimator.get_variable_names()
        if not FILTER_PRED(p)
    ]
    batched_norms = [
        np.linalg.norm(param, axis=1) / param.shape[1]
        for param in params
        if hasattr(param, "shape") and len(param.shape) == 2
    ]
    return [item for sublist in batched_norms for item in sublist]


def main(args):
    # Can look at both the histogram and the norm of the weights.
    ckpt_ids = []
    norms = []
    norms_by_layer = defaultdict(list)

    last_param, param = None, None
    last_param_layer, param_layer = [None for _ in range(12)], [None for _ in range(12)]
    dir_sims = []
    dir_sims_by_layer = defaultdict(list)

    for trial in range(1):
        model = MtfModel(f"/home/willm/data/bsl/bsl-{trial}/", tpu=None)
        gin.parse_config_file(_operative_config_path(model._model_dir))
        vocabulary = t5.data.get_mixture_or_task(MIXTURE_NAME).get_vocabulary()
        ckpts = get_checkpoints(model._model_dir)
        if args.samples is not None:
            ckpts = list(downsample(ckpts, samples=args.samples))

        for n, ckpt in enumerate(tqdm.tqdm(ckpts)):
            print(f"Starting ckpt {ckpt}...")
            ckpt_id = int(ckpt.split("-")[1])
            ckpt_ids.append(ckpt_id)
            write_checkpoint_file(ckpt)

            estimator = model.estimator(vocabulary, init_checkpoint=ckpt)
            param_names = get_param_names(estimator)

            values = (estimator.get_variable_value(p) for p in param_names)
            norm = get_param_norm(values, normalize=False)
            norms.append(norm)
            print(f"({n}/{len(ckpts)}) norm({ckpt_id}) = {norm:.0f}")

            for layer in range(12):
                layer_params = filter_by_layer(param_names, layer)
                values = (estimator.get_variable_value(p) for p in layer_params)
                norm = get_param_norm(values, normalize=False)
                norms_by_layer[layer].append(norm)
            
            last_param = param
            values = (estimator.get_variable_value(p) for p in param_names)
            param = get_param(values)
            for layer in range(12):
                last_param_layer[layer] = param_layer[layer]
                layer_params = filter_by_layer(param_names, layer)
                values = (estimator.get_variable_value(p) for p in layer_params)
                param_layer[layer] = get_param(values)

            if last_param is not None:
                dir_sim = (param @ last_param) / (norms[-1] * norms[-2])
                dir_sims.append(dir_sim)
                for layer, (param_, last_param_) in enumerate(zip(param_layer, last_param_layer)):
                    norm_ = norms_by_layer[layer][-1]
                    last_norm_ = norms_by_layer[layer][-2]
                    dir_sim_ = (param_ @ last_param_) / (norm_ * last_norm_)
                    dir_sims_by_layer[layer].append(dir_sim_)

    plt.figure()
    plt.plot(ckpt_ids, norms)
    plt.xlabel("Pretraining checkpoint")
    plt.ylabel("Parameter norm")
    plt.title("Aggregate parameter norm")
    plt.savefig(f"images/t5-norm.png")
    print("Saved t5-norm figure.")

    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 0.8, 12))

    plt.figure()
    for (layer, data), color in zip(norms_by_layer.items(), colors):
        plt.plot(ckpt_ids, data, label=f"Layer {layer + 1}", color=color)
    plt.xlabel("Pretraining checkpoint")
    plt.ylabel("Parameter norm")
    plt.title("Parameter norm by layer")
    plt.legend()
    plt.savefig(f"images/t5-norm-by-layer.png")
    print("Saved t5-norm-by-layer figure.")

    plt.figure()
    plt.plot(ckpt_ids[1:], dir_sims, ".")
    plt.xlabel("Checkpoint $t$")
    plt.ylabel(R"$\mathrm{cos}(\theta_{t-1}, \theta_{t})$")
    plt.title("Change in param direction")
    plt.savefig(f"images/t5-dir.pdf")
    print("Saved t5-dir figure.")

    plt.figure()
    for (layer, data), color in zip(dir_sims_by_layer.items(), colors):
        plt.plot(ckpt_ids[1:], data, ".", label=f"Layer {layer + 1}", color=color)
    plt.xlabel("Checkpoint $t$")
    plt.ylabel(R"$\mathrm{cos}(\theta_{t-1}, \theta_{t}$)")
    plt.title("Change in param direction by layer")
    plt.legend()
    plt.savefig(f"images/t5-dir-by-layer.pdf")
    print("Saved t5-dir-by-layer figure.")

    if args.no_save:
        return

    # Save the norm data, which is expensive to compute.
    with open(f"/home/willm/data/bsl/t5-deriv/norms.dat", "wb") as fh:
        pickle.dump(norms, fh)
    with open(f"/home/willm/data/bsl/t5-deriv/ckpts.dat", "wb") as fh:
        pickle.dump(ckpt_ids, fh)
    with open(f"/home/willm/data/bsl/t5-deriv/norms_by_layer.dat", "wb") as fh:
        pickle.dump(norms_by_layer, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=None, type=int)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    main(args)
