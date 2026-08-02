import unittest

import torch

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.explorative_modeling import (
    select_xm_candidate_losses,
    xm_best_of_k_forward,
)
from decifer.minicif_v2 import MinicifV2Tokenizer


class ExplorativeModelingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MinicifV2Tokenizer()
        ids = self.tokenizer.encode(
            self.tokenizer.tokenize_minicif("<mcif2> Na formula Na 1 ")
        )
        self.idx = torch.tensor([ids, ids])
        self.targets = self.idx.clone()
        self.targets[1, -2:] = -1
        self.starts = [[0], [0]]

    def model(self, best_of_k=1, dropout=0.0):
        return Decifer(DeciferConfig(
            tokenizer="minicif_v2",
            vocab_size=self.tokenizer.vocab_size,
            block_size=32,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=dropout,
            record_aligned_attention=True,
            xm_best_of_k=best_of_k,
        ))

    def test_k1_is_exact_baseline_forward(self):
        torch.manual_seed(7)
        model = self.model(best_of_k=1, dropout=0.25)
        model.train()
        rng_state = torch.random.get_rng_state()

        expected_logits, expected_loss = model(
            self.idx,
            targets=self.targets,
            start_indices_batch=self.starts,
        )
        torch.random.set_rng_state(rng_state)
        actual_logits, actual_loss = xm_best_of_k_forward(
            model,
            self.idx,
            None,
            self.targets,
            self.starts,
            1,
        )

        self.assertIsNone(model.xm_mode_embeddings)
        self.assertTrue(torch.equal(actual_logits, expected_logits))
        self.assertTrue(torch.equal(actual_loss, expected_loss))

        with self.assertRaisesRegex(ValueError, "xm_best_of_k > 1"):
            model(
                self.idx,
                targets=self.targets,
                start_indices_batch=self.starts,
                xm_mode_indices=torch.zeros(2, dtype=torch.long),
            )

    def test_xm_keeps_shared_parameter_initialization_and_optimizer_coverage(self):
        torch.manual_seed(29)
        baseline = self.model(best_of_k=1)
        torch.manual_seed(29)
        xm_model = self.model(best_of_k=2)

        xm_state = xm_model.state_dict()
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, xm_state[name]), name)
        optimizer = xm_model.configure_optimizers(
            0.1,
            1e-3,
            (0.9, 0.95),
        )
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIn(id(xm_model.xm_mode_embeddings.weight), optimized)

    def test_selection_is_per_sample_and_routes_only_winner_gradients(self):
        candidate_losses = torch.tensor(
            [[1.0, 4.0], [2.0, 1.0], [3.0, 2.0]],
            requires_grad=True,
        )
        loss, winners = select_xm_candidate_losses(
            candidate_losses,
            torch.tensor([2, 4]),
        )

        loss.backward()

        self.assertEqual(winners.tolist(), [0, 1])
        expected_grad = torch.tensor(
            [[2.0 / 6.0, 0.0], [0.0, 4.0 / 6.0], [0.0, 0.0]]
        )
        self.assertTrue(torch.allclose(candidate_losses.grad, expected_grad))

    def test_candidate_search_replays_shared_dropout(self):
        torch.manual_seed(11)
        model = self.model(best_of_k=3, dropout=0.3)
        model.train()
        with torch.no_grad():
            shared = model.xm_mode_embeddings.weight[0].clone()
            model.xm_mode_embeddings.weight.copy_(shared.expand_as(
                model.xm_mode_embeddings.weight
            ))
        candidate_modes = torch.tensor([
            [0, 0],
            [1, 1],
            [2, 2],
        ])

        _, _, selection = xm_best_of_k_forward(
            model,
            self.idx,
            None,
            self.targets,
            self.starts,
            3,
            candidate_mode_indices=candidate_modes,
            return_selection=True,
        )

        self.assertTrue(torch.equal(
            selection.candidate_losses[0],
            selection.candidate_losses[1],
        ))
        self.assertTrue(torch.equal(
            selection.candidate_losses[0],
            selection.candidate_losses[2],
        ))

    def test_model_gradients_reach_only_winning_modes(self):
        torch.manual_seed(17)
        model = self.model(best_of_k=3)
        candidate_modes = torch.tensor([
            [0, 0],
            [1, 1],
            [2, 2],
        ])

        _, loss, selection = xm_best_of_k_forward(
            model,
            self.idx,
            None,
            self.targets,
            self.starts,
            3,
            candidate_mode_indices=candidate_modes,
            return_selection=True,
        )
        loss.backward()

        winners = set(selection.winner_mode_indices.tolist())
        gradient = model.xm_mode_embeddings.weight.grad
        self.assertGreater(float(gradient[list(winners)].abs().sum()), 0.0)
        for mode in set(range(3)) - winners:
            self.assertEqual(float(gradient[mode].abs().sum()), 0.0)

    def test_xm_mode_is_preserved_by_cached_generation(self):
        torch.manual_seed(23)
        model = self.model(best_of_k=2)
        model.eval()
        mode = torch.tensor([1, 1])
        next_token = self.idx[:, -1:]

        _, cache, past_length = model._prefill(
            self.idx,
            None,
            self.starts,
            None,
            mode,
        )
        cached_logits, _ = model._decode_step(
            next_token,
            cache,
            past_length,
        )
        full_logits, _ = model(
            torch.cat((self.idx, next_token), dim=1),
            start_indices_batch=self.starts,
            xm_mode_indices=mode,
        )

        self.assertTrue(torch.allclose(
            cached_logits,
            full_logits[:, -1, :],
            atol=1e-6,
            rtol=1e-5,
        ))

    def test_xm_rejects_packed_attention(self):
        with self.assertRaisesRegex(ValueError, "record_aligned_attention"):
            Decifer(DeciferConfig(
                tokenizer="minicif_v2",
                vocab_size=self.tokenizer.vocab_size,
                block_size=32,
                n_layer=1,
                n_head=1,
                n_embd=16,
                xm_best_of_k=2,
            ))


if __name__ == "__main__":
    unittest.main()
