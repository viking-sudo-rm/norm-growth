import torch
from torch import nn

from .si_transformer import SiTransConfig, SiEncoder


transformers = {
    "si_softmax": {},
    "no_scale": {"biases": True, "post_ln": True},
    "control": {"biases": True, "post_ln": True, "scale_scores": True},
    "control_rel": {"biases": True, "post_ln": True, "scale_scores": True, "rel_embed": True},
    "bias": {"biases": True},
    "post_ln": {"post_ln": True},
    "bias_scaled": {"biases": True, "scale_scores": True},
    "bias_scaled_rel": {"biases": True, "scale_scores": True, "rel_embed": True},
    "si_softmax_scaled": {"scale_scores": True},
    "control_drop": {"biases": True, "post_ln": True, "scale_scores": True, "p_drop": .1},
    "si_drop": {"p_drop": .1},
    "bias_scaled_drop": {"biases": True, "scale_scores": True, "p_drop": .1},
    "bias_scaled_drop3": {"biases": True, "scale_scores": True, "p_drop": .3},
    "pre_norm": {"biases": True, "scale_scores": True},
}


class LanguageModel(nn.Module):
    def __init__(
        self, d_model, d_ff, d_vocab, seq_len, n_heads, n_layers, encoder_type, bias=True,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(d_vocab, d_model)
        self.pos_embed = torch.nn.Embedding(seq_len, d_model)

        # Note: Before we were using the PyTorch transformer, but that one doesn't have masking.
        if encoder_type == "vaswani":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff
            )
            # I believe that this encoder is masked by default.
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            config = SiTransConfig(
                d_vocab,
                d_model,
                d_ff,
                n_heads,
                n_layers,
                seq_len - 1,
                masked=True,
                **transformers[encoder_type],
            )
            self.encoder = SiEncoder(config)

        self.classifier = nn.Linear(d_model, d_vocab, bias=bias)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        pos = torch.arange(0, seq_len, device=token_ids.device)
        embeddings = self.embed(token_ids) + self.pos_embed(pos).unsqueeze(dim=0)
        encodings = self.encoder(embeddings)
        return encodings, self.classifier(encodings)
