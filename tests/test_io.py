import json
from dataclasses import dataclass

import yaml

from decifer.io import create_run_layout


@dataclass
class DummyConfig:
    name: str = "demo"
    count: int = 3


def test_create_run_layout_initializes_standard_files(tmp_path):
    layout = create_run_layout(
        tmp_path / "demo-run",
        "evaluate",
        DummyConfig(),
        metadata={"dataset_name": "toy"},
    )

    assert (tmp_path / "demo-run" / "artifacts").is_dir()
    assert (tmp_path / "demo-run" / "logs").is_dir()
    assert (tmp_path / "demo-run" / "predictions").is_dir()

    with open(layout.run_config_path, "r") as f:
        run_config = yaml.safe_load(f)
    with open(layout.metadata_path, "r") as f:
        metadata = json.load(f)
    with open(layout.metrics_path, "r") as f:
        metrics = json.load(f)

    assert run_config["workflow"] == "evaluate"
    assert run_config["config"] == {"name": "demo", "count": 3}
    assert metadata["dataset_name"] == "toy"
    assert metadata["workflow"] == "evaluate"
    assert metrics == {}


def test_run_layout_updates_metadata_and_metrics(tmp_path):
    layout = create_run_layout(tmp_path / "train-run", "train", {"seed": 7})

    layout.write_metadata({"checkpoint_path": "/tmp/ckpt.pt"})
    layout.write_metrics({"losses": [1.0, 0.5], "best": 0.5})

    with open(layout.metadata_path, "r") as f:
        metadata = json.load(f)
    with open(layout.metrics_path, "r") as f:
        metrics = json.load(f)

    assert metadata["checkpoint_path"] == "/tmp/ckpt.pt"
    assert metrics["best"] == 0.5
    assert metrics["losses"] == [1.0, 0.5]
