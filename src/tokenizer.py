import torch


class Tokenizer:
    def __init__(self):
        self.mapping = {}
        self.idx = 0

    def tokenize(self, token, static=False):
        if token not in self.mapping:
            if static:
                # This happens 3 times on Wikitext-2 (capitalization), and never on PTB.
                token = "<unk>"
            else:
                self.mapping[token] = self.idx
                self.idx += 1
        return self.mapping[token]

    def gen_tokens(self, path, device=None, **args):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and line[0] != "=":
                    tokens = line.split()
                    yield torch.tensor(
                        [self.tokenize(tok, **args) for tok in tokens], device=device
                    )

    @property
    def d_vocab(self):
        return len(self.mapping)
