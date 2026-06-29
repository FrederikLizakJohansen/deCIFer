import unittest

import torch

from decifer.pxrd import clamp_qmax_for_wavelength, discrete_to_continuous_xrd, max_q_for_wavelength, nyquist_qstep, q_range_to_two_theta_range


class PxrdTest(unittest.TestCase):
    def test_nyquist_qstep_uses_points_per_fwhm(self):
        self.assertEqual(nyquist_qstep(0.04, 4), 0.01)

    def test_qmax_is_clamped_below_wavelength_limit(self):
        wavelength = 1.5406
        max_q = max_q_for_wavelength(wavelength)

        self.assertAlmostEqual(clamp_qmax_for_wavelength(10.0, wavelength), 0.95 * max_q)
        self.assertEqual(clamp_qmax_for_wavelength(4.0, wavelength), 4.0)

    def test_q_range_to_two_theta_never_uses_singular_limit(self):
        wavelength = 1.5406
        qmax, two_theta_range = q_range_to_two_theta_range(0.0, 10.0, wavelength)

        self.assertLess(qmax, max_q_for_wavelength(wavelength))
        self.assertLess(two_theta_range[1], 180.0)

    def test_discrete_to_continuous_xrd_augmented_shape(self):
        batch_q = torch.tensor([[1.0, 2.0, 0.0], [1.5, 2.5, 3.0]], dtype=torch.float32)
        batch_iq = torch.tensor([[1.0, 0.5, 0.0], [0.8, 0.4, 0.2]], dtype=torch.float32)

        out = discrete_to_continuous_xrd(
            batch_q,
            batch_iq,
            qmin=0.0,
            qmax=4.0,
            nyquist_points_per_fwhm=4,
            fwhm_range=(0.04, 0.04),
            noise_range=None,
            intensity_scale_range=None,
            mask_prob=None,
            q_shift_range=(-0.01, 0.01),
            q_scale_range=(0.99, 1.01),
            peak_intensity_jitter_range=(0.9, 1.1),
            peak_dropout_prob=0.0,
            background_range=(0.0, 0.01),
            impurity_peak_count_range=(0, 1),
            impurity_intensity_range=(0.01, 0.02),
            particle_size_range=(20.0, 20.0),
            peak_asymmetry_range=(-0.1, 0.1),
            final_normalize=True,
        )

        self.assertEqual(out["iq"].shape, (2, 400))
        self.assertEqual(out["q"].shape, (400,))
        self.assertTrue(torch.isfinite(out["iq"]).all())
        self.assertGreaterEqual(float(out["iq"].min()), 0.0)
        self.assertLessEqual(float(out["iq"].max()), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
