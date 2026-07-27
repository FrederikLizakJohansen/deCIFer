import unittest

import torch
from braggcalculator import (
    CalibrationArtifacts,
    PeakProfileArtifacts,
    SimulationArtifacts,
    render_artifact_batch,
)

from decifer.pxrd import (
    BraggArtifactSpec,
    bragg_artifact_batch,
    clamp_qmax_for_wavelength,
    discrete_to_continuous_xrd,
    load_bragg_artifact_spec,
    max_q_for_wavelength,
    nyquist_qstep,
    q_range_to_two_theta_range,
)


class PxrdTest(unittest.TestCase):
    def test_bragg_peak_batch_restores_padding_after_position_shift(self):
        spec = BraggArtifactSpec(
            artifacts=SimulationArtifacts(
                calibration=CalibrationArtifacts(zero_shift=0.1),
                domain="q",
            )
        )

        result = bragg_artifact_batch(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            spec=spec,
            include_dense=False,
            include_peaks=True,
        )

        self.assertTrue(torch.allclose(result["peak_q"], torch.tensor([[1.1, 0.0]])))
        self.assertTrue(torch.equal(result["peak_iq"], torch.tensor([[1.0, 0.0]])))

    def test_bragg_hybrid_dense_branch_uses_same_shifted_peaks(self):
        artifacts = SimulationArtifacts(
            calibration=CalibrationArtifacts(zero_shift=0.2),
            profile=PeakProfileArtifacts(
                model="pseudo_voigt", fwhm=0.08, eta=0.4
            ),
            normalize_signal=True,
            final_normalize=True,
            domain="q",
        )
        result = bragg_artifact_batch(
            torch.tensor([[1.0, 2.0, 0.0]]),
            torch.tensor([[1.0, 0.5, 0.0]]),
            spec=BraggArtifactSpec(artifacts=artifacts),
            qmin=0.0,
            qmax=4.0,
            qstep=0.02,
            include_dense=True,
            include_peaks=True,
        )
        mask = result["peak_q"] != 0
        expected = render_artifact_batch(
            result["peak_q"],
            result["peak_iq"],
            peak_mask=mask,
            grid=result["q"],
            artifacts=SimulationArtifacts(
                profile=artifacts.profile,
                normalize_signal=True,
                final_normalize=True,
                domain="q",
            ),
            wavelength=1.5406,
        )

        self.assertTrue(torch.allclose(result["iq"], expected))

    def test_full_artifact_yaml_is_seeded_and_repeatable(self):
        spec = load_bragg_artifact_spec(
            "configs/xrd_artifacts/full_evaluation.yaml"
        )
        batch_q = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
        batch_iq = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)

        first = bragg_artifact_batch(
            batch_q,
            batch_iq,
            spec=spec,
            qmin=0.0,
            qmax=4.0,
            qstep=0.02,
            include_dense=True,
            include_peaks=True,
        )
        second = bragg_artifact_batch(
            batch_q,
            batch_iq,
            spec=spec,
            qmin=0.0,
            qmax=4.0,
            qstep=0.02,
            include_dense=True,
            include_peaks=True,
        )

        self.assertEqual(spec.artifacts.profile.model, "tch")
        self.assertEqual(spec.artifacts.seed, 2026)
        self.assertTrue(torch.equal(first["peak_q"], second["peak_q"]))
        self.assertTrue(torch.equal(first["iq"], second["iq"]))

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

    def test_discrete_to_continuous_xrd_can_cap_peaks(self):
        batch_q = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        batch_iq = torch.tensor([[1.0, 0.1, 0.9]], dtype=torch.float32)

        capped = discrete_to_continuous_xrd(
            batch_q,
            batch_iq,
            qmin=0.0,
            qmax=4.0,
            qstep=0.02,
            fwhm_range=(0.04, 0.04),
            eta_range=(0.5, 0.5),
            noise_range=None,
            intensity_scale_range=None,
            mask_prob=None,
            max_peaks=1,
        )
        expected = discrete_to_continuous_xrd(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            qmin=0.0,
            qmax=4.0,
            qstep=0.02,
            fwhm_range=(0.04, 0.04),
            eta_range=(0.5, 0.5),
            noise_range=None,
            intensity_scale_range=None,
            mask_prob=None,
        )

        self.assertTrue(torch.allclose(capped["iq"], expected["iq"]))


if __name__ == "__main__":
    unittest.main()
