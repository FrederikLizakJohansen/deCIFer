import json
import os
import tempfile
import unittest

import numpy as np
from braggcalculator import BraggCalculator, OptimizationStage, RefinementPolicy
from pymatgen.core import Lattice, Structure

from decifer.bragg_refinement import (
    BraggRefinementConfig,
    load_bragg_refinement_config,
    refinement_coordinate,
    run_bragg_refinement,
)


class BraggRefinementTest(unittest.TestCase):
    def test_repository_refinement_presets_load(self):
        config_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "refinement",
        )
        expected = {
            "quick.yaml": ("quick", False, 1, False),
            "cautious.yaml": ("cautious", False, 1, False),
            "robust.yaml": ("robust", False, 3, False),
            "cautious_coordinates.yaml": ("cautious", True, 1, False),
            "cautious_species_assignment.yaml": (
                "cautious",
                False,
                1,
                True,
            ),
        }

        for filename, values in expected.items():
            config = load_bragg_refinement_config(
                os.path.join(config_dir, filename),
                default_wavelength=1.5406,
            )
            actual = (
                config.policy_name,
                config.policy.refine_coordinates,
                config.policy.restarts,
                config.species_assignment is not None,
            )
            self.assertEqual(actual, values)

    def test_small_synthetic_q_refinement_returns_structured_result(self):
        reference = Structure(
            Lattice.cubic(5.43),
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        candidate = Structure(
            Lattice.cubic(5.50),
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        calculator = BraggCalculator(
            mode="xray",
            wavelength=1.5406,
            q_range=(0.5, 7.0),
            q_step=0.04,
            primitive=False,
        ).load(reference)
        q, observed = calculator.pattern(domain="q")
        policy = RefinementPolicy(
            background_degree=1,
            diagnostic_points=0,
            stages=(
                OptimizationStage(
                    "synthetic",
                    ("scale", "background", "lattice"),
                    2,
                    0.01,
                ),
            ),
        )
        config = BraggRefinementConfig(
            enabled=True,
            domain="q",
            radiation="xray",
            wavelength=1.5406,
            device="cpu",
            policy_name="quick",
            policy=policy,
        )

        result, record = run_bragg_refinement(
            q,
            observed,
            candidate,
            config,
            initial_fit_statistics={"evaluation_r_wp": 0.5},
        )

        self.assertIsNotNone(result)
        self.assertTrue(record["succeeded"])
        self.assertEqual(len(record["coordinate"]), len(q))
        self.assertEqual(len(record["calculated_profile"]), len(q))
        self.assertEqual(len(record["residual"]), len(q))
        self.assertIn("r_wp", record["refined_fit_statistics"])
        self.assertIn("_cell_length_a", record["refined_cif"])
        self.assertTrue(record["refined_parameters"])
        json.dumps(record, allow_nan=False)

    def test_configuration_supports_two_theta_and_species_assignment(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "refinement",
            "quick.yaml",
        )
        config = load_bragg_refinement_config(
            path,
            domain="two_theta",
            radiation="neutron",
            default_wavelength=1.8,
        )

        coordinate = refinement_coordinate(np.asarray([1.0, 2.0, 3.0]), config)

        self.assertTrue(config.enabled)
        self.assertEqual(config.domain, "two_theta")
        self.assertEqual(config.radiation, "neutron")
        self.assertTrue(np.all(np.diff(coordinate) > 0))
        self.assertIsNone(config.species_assignment)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "enabled: true\n"
                "species_assignment:\n"
                "  enabled: true\n"
                "  search: pairwise\n"
                "  max_candidates: 16\n"
                "  continuous_top_k: 2\n"
            )
            handle.flush()
            species_config = load_bragg_refinement_config(
                handle.name,
                default_wavelength=1.5406,
            )

        self.assertIsNotNone(species_config.species_assignment)
        self.assertEqual(species_config.species_assignment.search, "pairwise")

    def test_refinement_failure_is_returned_as_candidate_record(self):
        config = load_bragg_refinement_config(
            enabled=True,
            default_wavelength=1.5406,
        )

        def fail_refinement(**kwargs):
            raise ValueError("invalid candidate")

        result, record = run_bragg_refinement(
            [1.0, 2.0, 3.0],
            [1.0, 0.5, 0.2],
            object(),
            config,
            refine_fn=fail_refinement,
        )

        self.assertIsNone(result)
        self.assertFalse(record["succeeded"])
        self.assertEqual(record["error"]["type"], "ValueError")
        self.assertIn("invalid candidate", record["error"]["message"])


if __name__ == "__main__":
    unittest.main()
