import importlib.util
import os
import tempfile
import unittest

from omegaconf import OmegaConf
import torch
from torch.utils.data import SequentialSampler


MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "train.py")
spec = importlib.util.spec_from_file_location("train_module", MODULE_PATH)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
TokenBudgetBatchSampler = train_module.TokenBudgetBatchSampler
load_trusted_checkpoint = train_module.load_trusted_checkpoint


class TokenBudgetBatchSamplerTest(unittest.TestCase):
    def test_trusted_checkpoint_loads_omegaconf_state(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint_file:
            torch.save(
                {"config": OmegaConf.create({"layers": [2, 4]})},
                checkpoint_file.name,
            )

            checkpoint = load_trusted_checkpoint(
                checkpoint_file.name, map_location="cpu"
            )

        self.assertEqual(list(checkpoint["config"].layers), [2, 4])

    def test_batches_cover_records_without_exceeding_padded_budget(self):
        lengths = [9, 10, 19, 20, 29, 30]
        sampler = TokenBudgetBatchSampler(
            SequentialSampler(lengths),
            lengths,
            token_budget=64,
            max_batch_size=4,
            condition_tokens=2,
            bucket_size=6,
            seed=42,
        )

        batches = list(sampler)

        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(len(lengths))))
        for batch in batches:
            padded_tokens = max(lengths[index] - 1 + 2 for index in batch) * len(batch)
            self.assertLessEqual(padded_tokens, 64)
            self.assertLessEqual(len(batch), 4)

    def test_single_record_larger_than_budget_is_rejected(self):
        sampler = TokenBudgetBatchSampler(
            SequentialSampler([100]),
            [100],
            token_budget=64,
            max_batch_size=4,
        )

        with self.assertRaisesRegex(ValueError, "exceeds batch_token_budget"):
            list(sampler)


if __name__ == "__main__":
    unittest.main()
