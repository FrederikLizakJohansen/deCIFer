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

    def test_conv_condition_encoder_parameters_are_grouped_for_optimizer(self):
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

        optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95))

        self.assertEqual(len(optimizer.param_groups), 2)

    def test_condition_attention_mask_blocks_attention_between_packed_records(self):
        tokenizer = MinicifTokenizer()
        model = Decifer(DeciferConfig(
            tokenizer="minicif",
            vocab_size=tokenizer.vocab_size,
            block_size=32,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
            condition=True,
            condition_size=32,
            condition_encoder="conv",
            condition_n_tokens=2,
            pxrd_encoder_channels=8,
        ))
        tokens = tokenizer.tokenize_minicif("<mcif> Na </mcif> <mcif> Cl </mcif>")
        ids = tokenizer.encode(tokens)
        idx = torch.tensor([ids])
        start_id = tokenizer.token_to_id["<mcif>"]
        start_indices = [[i for i, token_id in enumerate(ids) if token_id == start_id]]
        cond = torch.randn(2, 32)

        model(idx, cond, idx.clone(), start_indices, return_attn=True)

        attention = model.attn_scores[0][0]
        first_record_positions = torch.arange(0, start_indices[0][1] + 2)
        second_record_last_position = attention.size(0) - 1
        blocked_attention = attention[second_record_last_position, first_record_positions]
        self.assertTrue(torch.all(blocked_attention == 0))


if __name__ == "__main__":
    unittest.main()
