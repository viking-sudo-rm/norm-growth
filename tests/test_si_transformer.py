from unittest import TestCase
from copy import copy

from src.si_transformer import *
from src.saturate import saturate


class TestSiTransformer(TestCase):

    def setUp(self):
        self.config = SiTransConfig(500, 120, 256, 12, 12, 512)
        self.token_ids = torch.randint(low=0, high=499, size=[16, 512])

    def test_self_attn_dims(self):
        attn = SiSelfAttention(self.config)
        out = attn(torch.ones(16, 512, 120))
        assert list(out.shape) == [16, 512, 10]

    def test_multihead_dims(self):
        heads = SiMultiHead(self.config)
        out = heads(torch.ones(16, 512, 120))
        assert list(out.shape) == [16, 512, 120]

    def test_ff_dims(self):
        ff = SiFeedforward(self.config)
        out = ff(torch.ones(16, 512, 120))
        assert list(out.shape) == [16, 512, 120]

    def test_scale_invariance(self):
        transformer = SiTransformer(self.config)
        encodings = transformer(self.token_ids)
        with saturate(transformer):
            sat_encodings = transformer(self.token_ids)

        # The output should be 1-homogeneous.
        torch.testing.assert_allclose(encodings, sat_encodings / 1000, rtol=1e-1, atol=1e-1)

    def test_masked_dims(self):
        config = SiTransConfig(500, 120, 256, 12, 12, 512, masked=True)
        transformer = SiTransformer(config)
        encodings = transformer(self.token_ids)
        assert list(encodings.shape) == [16, 512, 120]

    def test_softmax_dims(self):
        config = SiTransConfig(500, 120, 256, 12, 12, 512, softmax=True)
        transformer = SiTransformer(config)
        encodings = transformer(self.token_ids)
        assert list(encodings.shape) == [16, 512, 120]
