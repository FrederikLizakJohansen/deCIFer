import gzip
import json
import os
import tempfile
import unittest

from decifer.evaluation_checkpoint import (
    EvaluationCheckpoint,
    checkpoint_paths,
    export_refinement_records,
    export_split_metrics,
    load_combined_checkpoint_frame,
)


class EvaluationCheckpointTest(unittest.TestCase):
    def checkpoint_path(self, root, split):
        return os.path.join(
            root,
            "evaluation_checkpoints",
            f"{split}.sqlite3",
        )

    def test_round_trip_and_resume_completed_units(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.checkpoint_path(tmpdir, "test")
            checkpoint = EvaluationCheckpoint(path, {"model": "example"})
            checkpoint.save_unit(
                12,
                "pxrd-elements",
                [
                    {
                        "split": "test",
                        "sample_index": 12,
                        "prompt_mode": "pxrd-elements",
                        "rep": 0,
                        "rwp": 0.2,
                    },
                    {
                        "split": "test",
                        "sample_index": 12,
                        "prompt_mode": "pxrd-elements",
                        "rep": 1,
                        "rwp": 0.1,
                    },
                ],
                [],
            )
            checkpoint.close()

            resumed = EvaluationCheckpoint(path, {"model": "example"})
            self.assertEqual(
                resumed.completed_units(),
                {(12, "pxrd-elements")},
            )
            resumed.close()

            frame = load_combined_checkpoint_frame([path])
            self.assertEqual(frame["rep"].tolist(), [0, 1])
            self.assertEqual(frame["rwp"].tolist(), [0.2, 0.1])

    def test_signature_mismatch_requires_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.checkpoint_path(tmpdir, "val")
            EvaluationCheckpoint(path, {"num_reps": 4}).close()

            with self.assertRaisesRegex(
                ValueError,
                "--restart-splits",
            ):
                EvaluationCheckpoint(path, {"num_reps": 8})

            restarted = EvaluationCheckpoint(
                path,
                {"num_reps": 8},
                reset=True,
            )
            self.assertEqual(restarted.completed_units(), set())
            restarted.close()

    def test_failed_unit_write_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.checkpoint_path(tmpdir, "test")
            checkpoint = EvaluationCheckpoint(path, {"model": "example"})
            checkpoint.save_unit(
                1,
                "pxrd",
                [{"rep": 0, "sample_index": 1}],
                [],
            )

            with self.assertRaises(TypeError):
                checkpoint.save_unit(
                    2,
                    "pxrd",
                    [{"rep": 0, "unsupported": object()}],
                    [],
                )

            self.assertEqual(checkpoint.completed_units(), {(1, "pxrd")})
            checkpoint.close()

    def test_combines_splits_and_exports_refinement_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            refinement_record = {
                "split": "test",
                "sample_index": 2,
                "prompt_mode": "pxrd",
                "rep": 0,
                "objective_history": [0.4, 0.2],
                "refined_cif": "data_test",
            }
            for split, sample_index in (("train", 1), ("test", 2)):
                checkpoint = EvaluationCheckpoint(
                    self.checkpoint_path(tmpdir, split),
                    {"split": split},
                )
                checkpoint.save_unit(
                    sample_index,
                    "pxrd",
                    [{
                        "split": split,
                        "sample_index": sample_index,
                        "prompt_mode": "pxrd",
                        "rep": 0,
                    }],
                    [refinement_record] if split == "test" else [],
                )
                checkpoint.close()

            paths = checkpoint_paths(tmpdir)
            frame = load_combined_checkpoint_frame(paths)
            self.assertEqual(set(frame["split"]), {"train", "test"})

            export_split_metrics(paths, tmpdir)
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmpdir, "split_metrics", "train.csv")
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmpdir, "split_metrics", "test.csv")
                )
            )

            output_path = os.path.join(tmpdir, "refinement.jsonl.gz")
            self.assertEqual(
                export_refinement_records(paths, output_path),
                1,
            )
            with gzip.open(output_path, "rt", encoding="utf-8") as handle:
                exported = json.loads(handle.readline())
            self.assertEqual(exported, refinement_record)


if __name__ == "__main__":
    unittest.main()
