from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_sequence_to_len(sequence, seq_len):
    sequences = pad_sequence(sequence, batch_first=True)[:, : seq_len]
    if sequences.size(1) == seq_len:
        return sequences
    padding = torch.zeros(sequences.size(0), seq_len, device=sequences.device, dtype=torch.long)
    padding[:, : sequences.size(1)] = sequences
    return padding


def get_mask(tokens: List[torch.Tensor], seq_len):
    lengths = torch.tensor([len(sent) for sent in tokens], device=tokens[0].device)
    indices = (
        torch.arange(0, seq_len, device=lengths.device)
        .unsqueeze(dim=0)
        .repeat(len(lengths), 1)
    )
    lengths = lengths.unsqueeze(dim=-1)
    return torch.where(
        indices < lengths, torch.ones_like(indices), torch.zeros_like(indices)
    )

def get_text_tokens_mask(path, seq_len, tokenizer, name="data", log=None):
    if log:
        log.info(f"Loading train data from {path}...")
    raw_train = list(tokenizer.gen_tokens(path))
    train_tokens = pad_sequence_to_len(raw_train, args.seq_len)
    train_mask = get_mask(raw_train, args.seq_len).float()
    train_len = max(len(s) for s in raw_train)
    assert train_len <= args.seq_len
    if log:
        log.info(f"Max train sentence length is {train_len} (<= {args.seq_len}).")
    
    return raw_train, train_tokens, train_mask
