import unittest

import torch

from bin.pretrain_pxrd_encoder import SyntheticPxrdDataset, collate_fn, nt_xent_loss, pxrd_similarity_loss


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

    def test_collate_caps_raw_peaks_before_padding(self):
        batch = [
            {
                "xrd.q": torch.arange(6, dtype=torch.float32),
                "xrd.iq": torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
            },
            {
                "xrd.q": torch.arange(4, dtype=torch.float32),
                "xrd.iq": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            },
        ]

        collated = collate_fn(batch, max_raw_peaks_per_sample=3)

        self.assertEqual(collated["xrd.q"].shape, (2, 3))
        self.assertTrue(torch.all(collated["xrd.iq"][0] >= 0.7))

    def test_pxrd_similarity_loss_prefers_pxrd_neighbor_geometry(self):
        z1 = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        z2_good = z1.clone()
        z2_bad = torch.roll(z1, shifts=1, dims=0)
        dense_iq = torch.eye(4)

        good_loss = pxrd_similarity_loss(z1, z2_good, dense_iq, logit_temperature=0.1, target_temperature=0.1)
        bad_loss = pxrd_similarity_loss(z1, z2_bad, dense_iq, logit_temperature=0.1, target_temperature=0.1)

        self.assertLess(good_loss.item(), bad_loss.item())


if __name__ == "__main__":
    unittest.main()
