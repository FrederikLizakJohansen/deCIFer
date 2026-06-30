import unittest

import torch

from bin.pretrain_pxrd_encoder import nt_xent_loss


class PretrainPxrdEncoderTest(unittest.TestCase):
    def test_nt_xent_loss_prefers_matching_pairs(self):
        z1 = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        z2_good = z1.clone()
        z2_bad = torch.roll(z1, shifts=1, dims=0)

        good_loss = nt_xent_loss(z1, z2_good, temperature=0.1)
        bad_loss = nt_xent_loss(z1, z2_bad, temperature=0.1)

        self.assertLess(good_loss.item(), bad_loss.item())


if __name__ == "__main__":
    unittest.main()
