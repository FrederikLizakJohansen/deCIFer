import unittest

import torch

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.minicif import MinicifTokenizer


class DeciferModelTest(unittest.TestCase):
    def test_mlp_condition_encoder_keeps_single_condition_token(self):
        tokenizer = MinicifTokenizer()
        model = Decifer(DeciferConfig(
            tokenizer="minicif",
            vocab_size=tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=1,
            n_embd=16,
            condition=True,
            condition_size=32,
            condition_encoder="mlp",
            condition_n_tokens=1,
            condition_embedder_hidden_layers=[16],
        ))
        idx = torch.tensor([tokenizer.encode(tokenizer.tokenize_minicif("<mcif> Na "))])
        targets = idx.clone()
        cond = torch.randn(1, 32)

        logits, loss = model(idx, cond, targets, [[0]])

        self.assertEqual(logits.shape, (1, idx.size(1) + 1, tokenizer.vocab_size))
        self.assertIsNotNone(loss)

    def test_conv_condition_encoder_inserts_multiple_condition_tokens(self):
        tokenizer = MinicifTokenizer()
        model = Decifer(DeciferConfig(
            tokenizer="minicif",
            vocab_size=tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=1,
            n_embd=16,
            condition=True,
            condition_size=32,
            condition_encoder="conv",
            condition_n_tokens=4,
            pxrd_encoder_channels=8,
        ))
        idx = torch.tensor([tokenizer.encode(tokenizer.tokenize_minicif("<mcif> Na "))])
        targets = idx.clone()
        cond = torch.randn(1, 32)

        logits, loss = model(idx, cond, targets, [[0]])

        self.assertEqual(logits.shape, (1, idx.size(1) + 4, tokenizer.vocab_size))
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
