"""Scale-invariant modification of the transformer.

Can compare to the `BertModel` here:
https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel.
"""

from typing import NamedTuple
import torch
from torch import nn
from overrides import overrides
from math import sqrt, sin, cos

from src.saturate import saturate


class SiTransConfig(NamedTuple):
    """Wrapper object representing architectural hyperparameters."""
    n_vocab: int
    d_model: int
    d_hidden: int
    n_heads: int
    n_layers: int
    seq_len: int
    masked: bool = False
    softmax: bool = True
    biases: bool = False  # Add biases to the linear transformations.
    post_ln: bool = False  # Switch to post layer norm.
    scale_scores: bool = False  # Scale the attention weights by a sqrt factor.
    rel_embed: bool = False  # Should we used fixed relative positional embeddings instead of learned ones?
    p_drop: float = 0.  # Dropout probability.


class SiSelfAttention(nn.Module):
    def __init__(self, config: SiTransConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        d_head = config.d_model // config.n_heads
        self.query = nn.Linear(config.d_model, d_head, bias=config.biases)
        self.key = nn.Linear(config.d_model, d_head, bias=config.biases)
        self.value = nn.Linear(config.d_model, d_head, bias=config.biases)

        self.config = config
        self.d_head = d_head

        if not config.softmax:
            # Can't use standard LayerNorm here, since the mean shift will allow lookahead. This function will
            # produce a valid probability distribution.
            self.softmax = lambda x: x * x / x.norm(p=2, dim=-1, keepdim=True)
        else:
            self.softmax = nn.Softmax(dim=-1)

    @overrides
    def forward(self, encodings):
        queries = self.query(encodings)
        keys = self.key(encodings)
        values = self.value(encodings)
        scores = torch.einsum("bti, bsi -> bts", queries, keys)

        if self.config.scale_scores:
            scores = scores / sqrt(self.d_head)

        if self.config.masked:
            seq_len = scores.size(1)
            arange = torch.arange(seq_len, device=queries.device)
            mask = arange.unsqueeze(dim=0) <= arange.unsqueeze(dim=1)
            scores = mask.unsqueeze(dim=0) * scores

        weights = self.softmax(scores)
        return torch.einsum("bts, bsh -> bth", weights, values)


class SiMultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = [SiSelfAttention(config) for _ in range(config.n_heads)]
        for idx, head in enumerate(self.heads):
            self.add_module(f"head{idx}", head)
        
        self.pooler = torch.nn.Linear(config.d_model, config.d_model, bias=config.biases)
        self.lnorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.p_drop)
    
    @overrides
    def forward(self, encodings):
        heads = [head(encodings) for head in self.heads]
        outputs = self.pooler(torch.cat(heads, dim=-1))
        outputs = self.dropout(outputs)
        if not self.config.post_ln:
            return self.lnorm(outputs) + encodings
        else:
            return self.lnorm(outputs + encodings)


class SiFeedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_hidden = config.d_hidden
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden, bias=config.biases),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model, bias=config.biases),
        )
        self.lnorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(config.p_drop)
    
    @overrides
    def forward(self, encodings):
        outputs = self.net(encodings)
        outputs = self.dropout(outputs)
        if not self.config.post_ln:
            return self.lnorm(outputs) + encodings
        else:
            return self.lnorm(outputs + encodings)


class SiLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = SiMultiHead(config)
        self.ff = SiFeedforward(config)
    
    @overrides
    def forward(self, encodings):
        return self.ff(self.heads(encodings))


class SiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [SiLayer(config) for _ in range(config.n_layers)]
        for idx, layer in enumerate(self.layers):
            self.add_module(f"layer{idx}", layer)

    @overrides
    def forward(self, encodings):
        for layer in self.layers:
            encodings = layer(encodings)
        return encodings


class SiEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.n_vocab, config.d_model)
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)

        if config.rel_embed:
            # Relative positional embeddings taken from https://arxiv.org/pdf/1706.03762.pdf.
            self.pos_embed.requires_grad = False
            embeddings = self.pos_embed.weight
            for pos in range(config.seq_len):
                for idx in range(config.d_model // 2):
                    embeddings[pos, 2 * idx] = sin(pos / 10000**(2 * idx / config.d_model))
                    embeddings[pos, 2 * idx + 1] = cos(pos / 10000**(2 * idx / config.d_model))

    @overrides
    def forward(self, token_ids):
        _, seq_len = token_ids.size()
        embeddings = self.embed(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeddings = self.pos_embed(positions).unsqueeze(dim=0)
        return embeddings + pos_embeddings


class SiTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = SiEmbedder(config)
        self.encoder = SiEncoder(config)
    
    @overrides
    def forward(self, token_ids):
        embeddings = self.embedder(token_ids)
        return self.encoder(embeddings)
