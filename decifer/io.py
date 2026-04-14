#!/usr/bin/env python3

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yaml


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


@dataclass
class RunLayout:
    workflow_name: str
    root_dir: str
    artifacts_dir: str
    logs_dir: str
    predictions_dir: str
    run_config_path: str
    metadata_path: str
    metrics_path: str

    def write_metadata(self, metadata: Dict[str, Any], merge: bool = True) -> None:
        payload = {}
        if merge and os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                payload = json.load(f)
        payload.update(_to_serializable(metadata))
        with open(self.metadata_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        with open(self.metrics_path, "w") as f:
            json.dump(_to_serializable(metrics), f, indent=2, sort_keys=True)


def create_run_layout(
    root_dir: str,
    workflow_name: str,
    config: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> RunLayout:
    root_dir = os.path.abspath(root_dir)
    artifacts_dir = os.path.join(root_dir, "artifacts")
    logs_dir = os.path.join(root_dir, "logs")
    predictions_dir = os.path.join(root_dir, "predictions")

    for path in [root_dir, artifacts_dir, logs_dir, predictions_dir]:
        os.makedirs(path, exist_ok=True)

    layout = RunLayout(
        workflow_name=workflow_name,
        root_dir=root_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        predictions_dir=predictions_dir,
        run_config_path=os.path.join(root_dir, "run.yaml"),
        metadata_path=os.path.join(root_dir, "metadata.json"),
        metrics_path=os.path.join(root_dir, "metrics.json"),
    )

    run_payload = {
        "workflow": workflow_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _to_serializable(config),
    }
    with open(layout.run_config_path, "w") as f:
        yaml.safe_dump(run_payload, f, sort_keys=False)

    layout.write_metadata(
        {
            "workflow": workflow_name,
            "root_dir": root_dir,
            "created_at_utc": run_payload["created_at_utc"],
            **(metadata or {}),
        },
        merge=False,
    )
    layout.write_metrics({})
    return layout
