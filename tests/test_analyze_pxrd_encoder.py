import unittest

import numpy as np

from bin.analyze_pxrd_encoder import pearsonr, rankdata, rwp, spearmanr


class AnalyzePxrdEncoderTest(unittest.TestCase):
    def test_rankdata_handles_ties(self):
        ranks = rankdata(np.asarray([3.0, 1.0, 1.0, 2.0]))

        np.testing.assert_allclose(ranks, np.asarray([3.0, 0.5, 0.5, 2.0]))

    def test_correlations_and_rwp_are_well_behaved(self):
        x = np.asarray([1.0, 2.0, 3.0])
        y = np.asarray([2.0, 4.0, 6.0])

        self.assertAlmostEqual(pearsonr(x, y), 1.0)
        self.assertAlmostEqual(spearmanr(x, y), 1.0)
        self.assertAlmostEqual(rwp(x, x), 0.0)


if __name__ == "__main__":
    unittest.main()
