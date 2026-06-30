import unittest

import torch

from bin.pretrain_pxrd_encoder import SyntheticPxrdDataset, collate_fn, nt_xent_loss


class PretrainPxrdEncoderTest(unittest.TestCase):
    def test_nt_xent_loss_prefers_matching_pairs(self):
        z1 = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        z2_good = z1.clone()
        z2_bad = torch.roll(z1, shifts=1, dims=0)

        good_loss = nt_xent_loss(z1, z2_good, temperature=0.1)
        bad_loss = nt_xent_loss(z1, z2_bad, temperature=0.1)

        self.assertLess(good_loss.item(), bad_loss.item())

    def test_synthetic_pxrd_dataset_collates_without_hdf5(self):
        dataset = SyntheticPxrdDataset(size=4, qmin=0.0, qmax=10.0, seed=1337)
        batch = collate_fn([dataset[index] for index in range(4)])

        self.assertEqual(batch["xrd.q"].shape[0], 4)
        self.assertEqual(batch["xrd.iq"].shape, batch["xrd.q"].shape)
        self.assertTrue(torch.all(batch["xrd.q"] >= 0.0))
        self.assertTrue(torch.all(batch["xrd.q"] <= 10.0))


if __name__ == "__main__":
    unittest.main()
