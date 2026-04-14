#!/usr/bin/env python3

import argparse
import base64
import gzip
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import asdict, is_dataclass
from glob import glob
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "decifer-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from decifer.config import EvaluateConfig, TrainWorkflowConfig, load_dataclass_config

LOG_TAIL_LINES = 200
COMPARISON_LIMIT = 6

DEFAULT_MONITOR_SETTINGS = {
    "train": {
        "dataset": "",
        "out_dir": "runs/local_cpu_training",
        "init_from": "scratch",
        "device": "cpu",
        "dtype": "float32",
        "validate": True,
        "condition": True,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_iters": 100,
        "eval_interval": 10,
        "log_interval": 1,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 128,
        "block_size": 512,
        "learning_rate": 1e-3,
    },
    "eval": {
        "dataset_path": "",
        "dataset_split": "val",
        "model_ckpt": "",
        "out_folder": "runs/local_cpu_eval",
        "dataset_name": "val-preview",
        "model_name": "local-ui",
        "debug_max": 8,
        "num_reps": 1,
        "add_composition": True,
        "add_spacegroup": False,
        "max_new_tokens": 512,
        "temperature": 1.0,
        "condition": True,
        "override": True,
    },
}


def as_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): as_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(item) for item in value]
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


def load_training_config(config_path: str) -> TrainWorkflowConfig:
    return load_dataclass_config(TrainWorkflowConfig, config_path=config_path)


def load_evaluation_config(config_path: str) -> EvaluateConfig:
    return load_dataclass_config(EvaluateConfig, config_path=config_path)


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_training_run_dir_from_settings(settings: Dict[str, Any]) -> str:
    return os.path.abspath(settings["train"]["out_dir"])


def resolve_checkpoint_path_from_settings(settings: Dict[str, Any]) -> str:
    model_ckpt = settings["eval"].get("model_ckpt")
    if model_ckpt:
        return os.path.abspath(model_ckpt)
    return os.path.join(resolve_training_run_dir_from_settings(settings), "ckpt.pt")


def resolve_evaluation_run_dir_from_settings(settings: Dict[str, Any]) -> str:
    out_folder = settings["eval"].get("out_folder")
    if out_folder:
        return os.path.abspath(out_folder)
    checkpoint_path = resolve_checkpoint_path_from_settings(settings)
    base_dir = os.path.dirname(checkpoint_path) or "."
    dataset_name = settings["eval"].get("dataset_name", "default_dataset")
    model_name = settings["eval"].get("model_name", "default_model")
    return os.path.abspath(os.path.join(base_dir, "runs", "evaluate", f"{dataset_name}__{model_name}"))


def load_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"error": str(exc)}


def choose_native_path(kind: str, current_path: Optional[str] = None) -> Optional[str]:
    current_path = os.path.abspath(os.path.expanduser(current_path or str(REPO_ROOT)))
    initial_dir = current_path if os.path.isdir(current_path) else os.path.dirname(current_path)
    initial_dir = initial_dir or str(REPO_ROOT)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"Native file dialogs are unavailable: {exc}")

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        if kind == "file":
            selected = filedialog.askopenfilename(initialdir=initial_dir)
        elif kind == "dir":
            selected = filedialog.askdirectory(initialdir=initial_dir, mustexist=False)
        else:
            raise ValueError(f"Unsupported dialog kind: {kind}")
    finally:
        root.destroy()

    return os.path.abspath(selected) if selected else None


def residual_weighted_profile(sample: Any, generated: Any) -> Optional[float]:
    if sample is None or generated is None:
        return None
    sample = np.asarray(sample)
    generated = np.asarray(generated)
    if sample.size == 0 or generated.size == 0:
        return None
    denom = np.sum(np.square(sample))
    if denom == 0:
        return None
    return float(np.sqrt(np.sum(np.square(sample - generated)) / denom))


def plot_xrd_overlay(sample_q: Any, sample_iq: Any, gen_q: Any, gen_iq: Any) -> Optional[str]:
    if sample_q is None or sample_iq is None or gen_q is None or gen_iq is None:
        return None

    sample_q = np.asarray(sample_q)
    sample_iq = np.asarray(sample_iq)
    gen_q = np.asarray(gen_q)
    gen_iq = np.asarray(gen_iq)
    if sample_q.size == 0 or sample_iq.size == 0 or gen_q.size == 0 or gen_iq.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=120)
    ax.plot(sample_q, sample_iq, label="sample", linewidth=1.0, color="#111111")
    ax.plot(gen_q, gen_iq, label="generated", linewidth=1.0, color="#666666")
    ax.set_xlabel("Q")
    ax.set_ylabel("I(Q)")
    ax.grid(alpha=0.12)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_comparison_record(file_path: str) -> Dict[str, Any]:
    with gzip.open(file_path, "rb") as handle:
        row = pickle.load(handle)

    status = row.get("status", [])
    success = "success" in status
    validity = row.get("validity") or {}
    sample_xrd = row.get("xrd_clean_sample") or {}
    gen_xrd = row.get("xrd_clean_gen") or {}
    if not sample_xrd and row.get("xrd_q_continuous_sample") is not None and row.get("xrd_iq_continuous_sample") is not None:
        sample_xrd = {
            "q": row.get("xrd_q_continuous_sample"),
            "iq": row.get("xrd_iq_continuous_sample"),
        }
    plot_data_uri = None
    plot_error = row.get("xrd_error")

    try:
        plot_data_uri = plot_xrd_overlay(
            sample_xrd.get("q"),
            sample_xrd.get("iq"),
            gen_xrd.get("q"),
            gen_xrd.get("iq"),
        )
    except Exception as exc:
        plot_error = str(exc)
    if plot_data_uri is None and plot_error is None:
        plot_error = "Evaluation output did not contain plottable XRD arrays."

    return {
        "file_name": os.path.basename(file_path),
        "cif_name": row.get("cif_name"),
        "rep": row.get("rep"),
        "status": status,
        "success": success,
        "prompt_cif": row.get("prompt_string"),
        "prompt_flags": as_serializable(row.get("prompt_flags") or {}),
        "rmsd": as_serializable(row.get("rmsd")),
        "rwp": residual_weighted_profile(sample_xrd.get("iq"), gen_xrd.get("iq")),
        "validity": as_serializable(validity),
        "is_valid": bool(validity) and all(bool(v) for v in validity.values()),
        "sample_spacegroup": row.get("spacegroup_sample"),
        "generated_spacegroup": row.get("spacegroup"),
        "seq_len_sample": row.get("seq_len_sample"),
        "seq_len_gen": row.get("seq_len_gen"),
        "sample_cif": row.get("cif_string_sample"),
        "generated_completion_raw": row.get("cif_string_completion_raw"),
        "generated_cif_raw": row.get("cif_string_gen_raw"),
        "generated_cif": row.get("cif_string_gen"),
        "error_msg": row.get("error_msg"),
        "plot_data_uri": plot_data_uri,
        "plot_error": plot_error,
        "xrd_overlay_ready": bool(row.get("xrd_overlay_ready")),
    }


def load_validation_comparisons(run_dir: str, limit: int = COMPARISON_LIMIT) -> List[Dict[str, Any]]:
    pattern = os.path.join(run_dir, "predictions", "eval_files", "*", "*.pkl.gz")
    files = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
    comparisons = []
    for file_path in files[:limit]:
        try:
            comparisons.append(build_comparison_record(file_path))
        except Exception as exc:
            comparisons.append(
                {
                    "file_name": os.path.basename(file_path),
                    "success": False,
                    "status": ["error"],
                    "error_msg": str(exc),
                }
            )
    return comparisons


class ManagedProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.process: Optional[subprocess.Popen] = None
        self.command: List[str] = []
        self.cwd: Optional[str] = None
        self.output_lines: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self, command: List[str], cwd: str) -> None:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError(f"{self.name} is already running")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.command = command
            self.cwd = cwd
            self.output_lines.clear()
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._reader_thread = threading.Thread(target=self._drain_output, daemon=True)
            self._reader_thread.start()

    def _drain_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.output_lines.append(line.rstrip())

    def stop(self) -> None:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                return
            self.process.terminate()

    def snapshot(self) -> Dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        return {
            "running": running,
            "returncode": None if self.process is None else self.process.poll(),
            "pid": None if self.process is None else self.process.pid,
            "command": self.command,
            "cwd": self.cwd,
            "log_tail": list(self.output_lines),
        }


class MonitorApplication:
    def __init__(
        self,
        train_config_path: Optional[str],
        eval_config_path: Optional[str],
        python_executable: str,
        state_path: Optional[str] = None,
    ) -> None:
        self.train_config_path = os.path.abspath(train_config_path) if train_config_path else None
        self.eval_config_path = os.path.abspath(eval_config_path) if eval_config_path else None
        self.python_executable = python_executable
        self.state_path = os.path.abspath(state_path or os.path.join(REPO_ROOT, "runs", "local_monitor_state.json"))
        self.runtime_dir = os.path.dirname(self.state_path)
        os.makedirs(self.runtime_dir, exist_ok=True)

        self.train_process = ManagedProcess("training")
        self.eval_process = ManagedProcess("evaluation")
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        settings = deep_merge({}, DEFAULT_MONITOR_SETTINGS)
        if self.train_config_path:
            settings["train"] = deep_merge(settings["train"], asdict(load_training_config(self.train_config_path)))
        if self.eval_config_path:
            settings["eval"] = deep_merge(settings["eval"], asdict(load_evaluation_config(self.eval_config_path)))
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as handle:
                settings = deep_merge(settings, json.load(handle))
        self._save_settings(settings)
        return settings

    def _save_settings(self, settings: Optional[Dict[str, Any]] = None) -> None:
        payload = as_serializable(settings or self.settings)
        with open(self.state_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def update_settings(self, updates: Dict[str, Any]) -> None:
        self.settings = deep_merge(self.settings, updates)
        self._save_settings()

    def training_run_dir(self) -> str:
        return resolve_training_run_dir_from_settings(self.settings)

    def checkpoint_path(self) -> str:
        return resolve_checkpoint_path_from_settings(self.settings)

    def evaluation_run_dir(self) -> str:
        return resolve_evaluation_run_dir_from_settings(self.settings)

    def _train_config_dict(self) -> Dict[str, Any]:
        base = asdict(load_training_config(self.train_config_path)) if self.train_config_path else asdict(TrainWorkflowConfig())
        return deep_merge(base, self.settings["train"])

    def _eval_config_dict(self) -> Dict[str, Any]:
        base = asdict(load_evaluation_config(self.eval_config_path)) if self.eval_config_path else asdict(EvaluateConfig())
        config = deep_merge(base, self.settings["eval"])
        config["model_ckpt"] = self.checkpoint_path()
        return config

    def generated_train_config_path(self) -> str:
        return os.path.join(self.runtime_dir, "local_monitor_train.generated.yaml")

    def generated_eval_config_path(self) -> str:
        return os.path.join(self.runtime_dir, "local_monitor_eval.generated.yaml")

    def _write_generated_configs(self) -> None:
        with open(self.generated_train_config_path(), "w") as handle:
            yaml.safe_dump(as_serializable(self._train_config_dict()), handle, sort_keys=False)
        with open(self.generated_eval_config_path(), "w") as handle:
            yaml.safe_dump(as_serializable(self._eval_config_dict()), handle, sort_keys=False)

    def train_command(self) -> List[str]:
        self._write_generated_configs()
        return [self.python_executable, "bin/train.py", "--config", self.generated_train_config_path()]

    def evaluate_command(self) -> List[str]:
        self._write_generated_configs()
        return [self.python_executable, "bin/evaluate.py", "--config", self.generated_eval_config_path()]

    def start_training(self) -> None:
        if not self.settings["train"].get("dataset"):
            raise ValueError("Set a training dataset path first")
        self.train_process.start(self.train_command(), str(REPO_ROOT))

    def stop_training(self) -> None:
        self.train_process.stop()

    def start_evaluation(self) -> None:
        if not self.settings["eval"].get("dataset_path"):
            raise ValueError("Set an evaluation dataset path first")
        if not os.path.exists(self.checkpoint_path()):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path()}")
        self.eval_process.start(self.evaluate_command(), str(REPO_ROOT))

    def stop_evaluation(self) -> None:
        self.eval_process.stop()

    def snapshot(self) -> Dict[str, Any]:
        train_run_dir = self.training_run_dir()
        eval_run_dir = self.evaluation_run_dir()
        return {
            "settings": as_serializable(self.settings),
            "paths": {
                "repo_root": str(REPO_ROOT),
                "state_file": self.state_path,
                "generated_train_config": self.generated_train_config_path(),
                "generated_eval_config": self.generated_eval_config_path(),
                "train_run_dir": train_run_dir,
                "eval_run_dir": eval_run_dir,
                "checkpoint_path": self.checkpoint_path(),
                "checkpoint_exists": os.path.exists(self.checkpoint_path()),
            },
            "training": {
                **self.train_process.snapshot(),
                "metrics": load_json_file(os.path.join(train_run_dir, "metrics.json")),
                "metadata": load_json_file(os.path.join(train_run_dir, "metadata.json")),
            },
            "evaluation": {
                **self.eval_process.snapshot(),
                "metrics": load_json_file(os.path.join(eval_run_dir, "metrics.json")),
                "metadata": load_json_file(os.path.join(eval_run_dir, "metadata.json")),
                "comparisons": load_validation_comparisons(eval_run_dir),
            },
        }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>deCIFer Local Monitor</title>
  <style>
    :root {
      --bg: #ffffff;
      --panel: #ffffff;
      --line: #e5e7eb;
      --muted: #6b7280;
      --text: #111827;
      --soft: #f9fafb;
      --button: #111827;
      --button-text: #ffffff;
      --danger: #dc2626;
      --ok: #16a34a;
      --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }
    .wrap {
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }
    h1, h2, h3 { margin: 0 0 12px; font-weight: 600; }
    p { margin: 0; color: var(--muted); }
    .header {
      margin-bottom: 20px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }
    .layout {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 20px;
    }
    .sidebar, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .section {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .section:last-child { border-bottom: 0; }
    .stack { display: grid; gap: 10px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    .path-input {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    .path-input button {
      padding: 10px 12px;
      background: var(--soft);
      color: var(--text);
      border: 1px solid var(--line);
    }
    input[type=checkbox] {
      width: auto;
      padding: 0;
    }
    .checkbox {
      display: flex;
      align-items: center;
      gap: 8px;
      text-transform: none;
      letter-spacing: 0;
      font-size: 14px;
      color: var(--text);
    }
    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 8px;
      padding: 10px 14px;
      background: var(--button);
      color: var(--button-text);
      font: inherit;
      cursor: pointer;
    }
    button.secondary {
      background: var(--soft);
      color: var(--text);
      border: 1px solid var(--line);
    }
    button.stop {
      background: #ffffff;
      color: var(--danger);
      border: 1px solid #fecaca;
    }
    button:disabled { opacity: 0.5; cursor: default; }
    .main {
      display: grid;
      gap: 20px;
    }
    .panels {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    .panel-body {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #d1d5db;
    }
    .dot.running { background: var(--ok); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
    }
    .metric-label {
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }
    .metric-value {
      font-size: 18px;
      font-weight: 600;
    }
    pre {
      margin: 0;
      padding: 12px;
      min-height: 180px;
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
    }
    .paths {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .path-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
      font-size: 13px;
    }
    .path-card code {
      display: block;
      margin-top: 6px;
      font-family: var(--mono);
      word-break: break-all;
    }
    .comparisons {
      display: grid;
      gap: 12px;
    }
    .comparison-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 14px;
    }
    .comparison-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .tag {
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }
    .comparison-grid {
      display: grid;
      grid-template-columns: 300px 1fr 1fr;
      gap: 10px;
    }
    .comparison-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }
    .comparison-grid img {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .code-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--soft);
    }
    .code-box pre {
      min-height: 260px;
      max-height: 320px;
      background: transparent;
      border: 0;
      padding: 0;
    }
    .message {
      font-size: 13px;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .layout, .panels, .comparison-grid, .paths, .grid, .metrics {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>deCIFer Local Monitor</h1>
      <p>Set dataset and checkpoint paths here, run training locally, and inspect validation 1:1 comparisons.</p>
    </div>

    <div class="layout">
      <aside class="sidebar">
        <div class="section">
          <h2>Training Settings</h2>
          <div class="stack">
            <label>Dataset Root<div class="path-input"><input id="train-dataset" readonly /><button type="button" onclick="browseNativePath('train.dataset', 'dir')">Browse</button></div></label>
            <label>Output Dir<div class="path-input"><input id="train-out-dir" readonly /><button type="button" onclick="browseNativePath('train.out_dir', 'dir')">Browse</button></div></label>
            <div class="grid">
              <label>Init<select id="train-init-from"><option value="scratch">scratch</option><option value="resume">resume</option></select></label>
              <label>Device<select id="train-device"><option value="cpu">cpu</option><option value="cuda">cuda</option></select></label>
            </div>
            <div class="grid">
              <label>Dtype<select id="train-dtype"><option value="float32">float32</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option></select></label>
              <label>Batch Size<input id="train-batch-size" type="number" min="1" /></label>
            </div>
            <div class="grid">
              <label>Grad Accum<input id="train-grad-accum" type="number" min="1" /></label>
              <label>Max Iters<input id="train-max-iters" type="number" min="1" /></label>
            </div>
            <div class="grid">
              <label>Eval Interval<input id="train-eval-interval" type="number" min="1" /></label>
              <label>Log Interval<input id="train-log-interval" type="number" min="1" /></label>
            </div>
            <div class="grid">
              <label>Layers<input id="train-n-layer" type="number" min="1" /></label>
              <label>Heads<input id="train-n-head" type="number" min="1" /></label>
            </div>
            <div class="grid">
              <label>Embedding<input id="train-n-embd" type="number" min="1" /></label>
              <label>Block Size<input id="train-block-size" type="number" min="1" /></label>
            </div>
            <label>Learning Rate<input id="train-learning-rate" type="number" step="0.000001" min="0" /></label>
            <label class="checkbox"><input id="train-validate" type="checkbox" /> Validate during training</label>
            <label class="checkbox"><input id="train-condition" type="checkbox" /> Use XRD conditioning</label>
          </div>
        </div>
        <div class="section">
          <h2>Evaluation Settings</h2>
          <div class="stack">
            <label>Dataset Path<div class="path-input"><input id="eval-dataset-path" readonly /><button type="button" onclick="browseNativePath('eval.dataset_path', 'dir')">Browse</button></div></label>
            <div class="grid">
              <label>Split<select id="eval-dataset-split"><option value="train">train</option><option value="val">val</option><option value="test">test</option></select></label>
              <label>Debug Max<input id="eval-debug-max" type="number" min="1" /></label>
            </div>
            <label>Checkpoint Path<div class="path-input"><input id="eval-model-ckpt" readonly placeholder="Leave empty to use training out_dir/ckpt.pt" /><button type="button" onclick="browseNativePath('eval.model_ckpt', 'file')">Browse</button></div></label>
            <label>Eval Output Dir<div class="path-input"><input id="eval-out-folder" readonly /><button type="button" onclick="browseNativePath('eval.out_folder', 'dir')">Browse</button></div></label>
            <div class="grid">
              <label>Dataset Name<input id="eval-dataset-name" /></label>
              <label>Model Name<input id="eval-model-name" /></label>
            </div>
            <div class="grid">
              <label>Num Reps<input id="eval-num-reps" type="number" min="1" /></label>
              <label>Max New Tokens<input id="eval-max-new-tokens" type="number" min="1" /></label>
            </div>
            <label>Temperature<input id="eval-temperature" type="number" step="0.01" min="0" /></label>
            <label class="checkbox"><input id="eval-add-composition" type="checkbox" /> Add composition prompt</label>
            <label class="checkbox"><input id="eval-add-spacegroup" type="checkbox" /> Add spacegroup prompt</label>
            <label class="checkbox"><input id="eval-condition" type="checkbox" /> Use conditioning</label>
            <label class="checkbox"><input id="eval-override" type="checkbox" /> Override existing eval files</label>
          </div>
        </div>
        <div class="section">
          <div class="buttons">
            <button onclick="saveSettings()">Save Settings</button>
            <button class="secondary" onclick="refresh()">Reload State</button>
          </div>
          <p id="save-message" class="message"></p>
        </div>
      </aside>

      <main class="main">
        <section class="panel">
          <div class="section">
            <h2>Paths</h2>
            <div class="paths" id="paths"></div>
          </div>
        </section>

        <div class="panels">
          <section class="panel">
            <div class="panel-body">
              <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                <h2>Training</h2>
                <div class="buttons">
                  <button id="train-start" onclick="startTraining()">Start</button>
                  <button id="train-stop" class="stop" onclick="postAction('/api/train/stop')">Stop</button>
                </div>
              </div>
              <div id="train-status" class="status"><span class="dot"></span><span>idle</span></div>
              <div class="metrics" id="train-metrics"></div>
              <pre id="train-log"></pre>
            </div>
          </section>

          <section class="panel">
            <div class="panel-body">
              <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                <h2>Validation Preview</h2>
                <div class="buttons">
                  <button id="eval-start" onclick="startEvaluation()">Run</button>
                  <button id="eval-stop" class="stop" onclick="postAction('/api/eval/stop')">Stop</button>
                </div>
              </div>
              <div id="eval-status" class="status"><span class="dot"></span><span>idle</span></div>
              <div class="metrics" id="eval-metrics"></div>
              <pre id="eval-log"></pre>
            </div>
          </section>
        </div>

        <section class="panel">
          <div class="section">
            <h2>Validation Comparisons</h2>
            <div id="comparisons" class="comparisons"></div>
          </div>
        </section>
      </main>
    </div>
  </div>

  <script>
    const TRAIN_FIELDS = {
      dataset: {id: "train-dataset", type: "string"},
      out_dir: {id: "train-out-dir", type: "string"},
      init_from: {id: "train-init-from", type: "string"},
      device: {id: "train-device", type: "string"},
      dtype: {id: "train-dtype", type: "string"},
      validate: {id: "train-validate", type: "bool"},
      condition: {id: "train-condition", type: "bool"},
      batch_size: {id: "train-batch-size", type: "int"},
      gradient_accumulation_steps: {id: "train-grad-accum", type: "int"},
      max_iters: {id: "train-max-iters", type: "int"},
      eval_interval: {id: "train-eval-interval", type: "int"},
      log_interval: {id: "train-log-interval", type: "int"},
      n_layer: {id: "train-n-layer", type: "int"},
      n_head: {id: "train-n-head", type: "int"},
      n_embd: {id: "train-n-embd", type: "int"},
      block_size: {id: "train-block-size", type: "int"},
      learning_rate: {id: "train-learning-rate", type: "float"},
    };

    const EVAL_FIELDS = {
      dataset_path: {id: "eval-dataset-path", type: "string"},
      dataset_split: {id: "eval-dataset-split", type: "string"},
      model_ckpt: {id: "eval-model-ckpt", type: "string"},
      out_folder: {id: "eval-out-folder", type: "string"},
      dataset_name: {id: "eval-dataset-name", type: "string"},
      model_name: {id: "eval-model-name", type: "string"},
      debug_max: {id: "eval-debug-max", type: "optional-int"},
      num_reps: {id: "eval-num-reps", type: "int"},
      max_new_tokens: {id: "eval-max-new-tokens", type: "int"},
      temperature: {id: "eval-temperature", type: "float"},
      add_composition: {id: "eval-add-composition", type: "bool"},
      add_spacegroup: {id: "eval-add-spacegroup", type: "bool"},
      condition: {id: "eval-condition", type: "bool"},
      override: {id: "eval-override", type: "bool"},
    };

    const ALL_FIELDS = [
      ...Object.values(TRAIN_FIELDS),
      ...Object.values(EVAL_FIELDS),
    ];
    let saveTimer = null;
    let formDirty = false;
    let lastComparisonsSignature = null;

    function fmt(value) {
      if (value === null || value === undefined || value === "") return "—";
      if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(4) : String(value);
      return String(value);
    }

    function getFieldValue(descriptor) {
      const el = document.getElementById(descriptor.id);
      if (descriptor.type === "bool") return el.checked;
      if (descriptor.type === "int") return parseInt(el.value, 10);
      if (descriptor.type === "float") return parseFloat(el.value);
      if (descriptor.type === "optional-int") return el.value === "" ? null : parseInt(el.value, 10);
      return el.value.trim();
    }

    function setFieldValue(descriptor, value) {
      const el = document.getElementById(descriptor.id);
      if (descriptor.type === "bool") {
        el.checked = Boolean(value);
      } else if (descriptor.type === "optional-int") {
        el.value = value === null || value === undefined ? "" : value;
      } else {
        el.value = value ?? "";
      }
    }

    async function persistField(fieldKey, pathValue) {
      const [section, key] = fieldKey.split(".");
      const result = await postAction("/api/settings", {[section]: {[key]: pathValue}});
      document.getElementById("save-message").textContent = result.ok ? "Path saved." : "Save failed.";
      return result;
    }

    async function browseNativePath(fieldKey, kind) {
      const [section, key] = fieldKey.split(".");
      const sectionFields = section === "train" ? TRAIN_FIELDS : EVAL_FIELDS;
      const descriptor = sectionFields[key];
      const currentValue = document.getElementById(descriptor.id).value.trim();
      const result = await postAction("/api/dialog", {field_key: fieldKey, kind, current_value: currentValue});
      if (!result.ok) {
        document.getElementById("save-message").textContent = "Selection failed.";
        return;
      }
      if (result.cancelled) {
        document.getElementById("save-message").textContent = "Selection cancelled.";
        return;
      }
      document.getElementById("save-message").textContent = "Path saved.";
      await refresh();
    }

    function fillSettingsForm(settings, force=false) {
      for (const [key, descriptor] of Object.entries(TRAIN_FIELDS)) {
        const el = document.getElementById(descriptor.id);
        if (force || document.activeElement !== el) setFieldValue(descriptor, settings.train[key]);
      }
      for (const [key, descriptor] of Object.entries(EVAL_FIELDS)) {
        const el = document.getElementById(descriptor.id);
        if (force || document.activeElement !== el) setFieldValue(descriptor, settings.eval[key]);
      }
    }

    function collectSettings() {
      const train = {};
      const evalSettings = {};
      for (const [key, descriptor] of Object.entries(TRAIN_FIELDS)) train[key] = getFieldValue(descriptor);
      for (const [key, descriptor] of Object.entries(EVAL_FIELDS)) evalSettings[key] = getFieldValue(descriptor);
      return {train, eval: evalSettings};
    }

    function scheduleAutoSave() {
      formDirty = true;
      document.getElementById("save-message").textContent = "Unsaved changes.";
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {
        saveSettings({refreshAfter: false, successMessage: "Settings auto-saved."});
      }, 400);
    }

    function attachAutoSaveHandlers() {
      for (const descriptor of ALL_FIELDS) {
        const el = document.getElementById(descriptor.id);
        const eventName = descriptor.type === "bool" ? "change" : "input";
        el.addEventListener(eventName, scheduleAutoSave);
        if (descriptor.type !== "bool") {
          el.addEventListener("change", scheduleAutoSave);
        }
      }
    }

    function metricCard(label, value) {
      return `<div class="metric"><span class="metric-label">${label}</span><div class="metric-value">${fmt(value)}</div></div>`;
    }

    function renderMetrics(containerId, metrics, fields) {
      document.getElementById(containerId).innerHTML = fields.map(([label, key]) => metricCard(label, metrics ? metrics[key] : null)).join("");
    }

    function renderPaths(paths) {
      const entries = [
        ["Train Run Dir", paths.train_run_dir],
        ["Eval Run Dir", paths.eval_run_dir],
        ["Checkpoint", paths.checkpoint_path],
        ["State File", paths.state_file],
        ["Generated Train Config", paths.generated_train_config],
        ["Generated Eval Config", paths.generated_eval_config],
      ];
      document.getElementById("paths").innerHTML = entries.map(([label, value]) =>
        `<div class="path-card"><strong>${label}</strong><code>${value}</code></div>`
      ).join("");
    }

    function setStatus(elementId, running, extraText="") {
      const el = document.getElementById(elementId);
      const dotClass = running ? "dot running" : "dot";
      const label = running ? `running ${extraText}` : `stopped ${extraText}`;
      el.innerHTML = `<span class="${dotClass}"></span><span>${label}</span>`;
    }

    function renderComparisons(comparisons) {
      const container = document.getElementById("comparisons");
      const signature = JSON.stringify(comparisons || []);
      if (signature === lastComparisonsSignature) return;
      lastComparisonsSignature = signature;
      if (!comparisons || comparisons.length === 0) {
        container.innerHTML = `<p class="message">No validation comparisons yet.</p>`;
        return;
      }
      container.innerHTML = comparisons.map((item) => {
        const tags = [
          `<span class="tag">success: ${item.success}</span>`,
          `<span class="tag">valid: ${item.is_valid}</span>`,
          `<span class="tag">rmsd: ${fmt(item.rmsd)}</span>`,
          `<span class="tag">rwp: ${fmt(item.rwp)}</span>`,
          `<span class="tag">sample sg: ${fmt(item.sample_spacegroup)}</span>`,
          `<span class="tag">gen sg: ${fmt(item.generated_spacegroup)}</span>`,
          `<span class="tag">overlay: ${item.xrd_overlay_ready}</span>`,
        ].join("");
        const image = item.plot_data_uri
          ? `<img src="${item.plot_data_uri}" alt="XRD overlay" />`
          : `<div class="code-box"><p class="message">${item.plot_error ? `XRD overlay failed: ${item.plot_error}` : "No XRD overlay available."}</p></div>`;
        const promptFlags = item.prompt_flags || {};
        return `
          <div class="comparison-card">
            <div class="comparison-head">
              <div><strong>${item.cif_name || item.file_name}</strong> rep ${fmt(item.rep)}</div>
              <div class="tags">${tags}</div>
            </div>
            <div class="comparison-meta">
              <div class="code-box">
                <strong>Prompt</strong>
                <pre>${(item.prompt_cif || "").replace(/</g, "&lt;")}</pre>
              </div>
              <div class="code-box">
                <strong>Model Completion</strong>
                <pre>${(item.generated_completion_raw || "").replace(/</g, "&lt;")}</pre>
              </div>
            </div>
            <div class="comparison-grid">
              ${image}
              <div class="code-box">
                <strong>Sample CIF</strong>
                <pre>${(item.sample_cif || "").replace(/</g, "&lt;")}</pre>
              </div>
              <div class="code-box">
                <strong>Generated Full Sequence</strong>
                <pre>${(item.generated_cif || item.generated_cif_raw || item.error_msg || "").replace(/</g, "&lt;")}</pre>
              </div>
            </div>
            <div class="comparison-meta" style="margin-top:10px;">
              <div class="code-box">
                <strong>Prompt Flags</strong>
                <pre>${JSON.stringify(promptFlags, null, 2).replace(/</g, "&lt;")}</pre>
              </div>
            </div>
          </div>
        `;
      }).join("");
    }

    async function postAction(path, body=null) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: body ? JSON.stringify(body) : null,
      });
      const payload = await response.json();
      if (!response.ok) alert(payload.error || "Request failed");
      return payload;
    }

    async function saveSettings(options={}) {
      const refreshAfter = options.refreshAfter ?? true;
      const successMessage = options.successMessage ?? "Settings saved.";
      const failureMessage = options.failureMessage ?? "Save failed.";
      if (saveTimer) {
        clearTimeout(saveTimer);
        saveTimer = null;
      }
      const payload = collectSettings();
      const result = await postAction("/api/settings", payload);
      if (result.ok) {
        formDirty = false;
        document.getElementById("save-message").textContent = successMessage;
      } else {
        document.getElementById("save-message").textContent = failureMessage;
      }
      if (refreshAfter) await refresh();
      return result;
    }

    async function startTraining() {
      const result = await saveSettings({refreshAfter: false, successMessage: "Settings saved. Starting training..."});
      if (!result.ok) return;
      await postAction("/api/train/start");
      await refresh();
    }

    async function startEvaluation() {
      const result = await saveSettings({refreshAfter: false, successMessage: "Settings saved. Starting evaluation..."});
      if (!result.ok) return;
      await postAction("/api/eval/start");
      await refresh();
    }

    async function refresh() {
      const response = await fetch("/api/state");
      const state = await response.json();
      if (!formDirty) fillSettingsForm(state.settings);
      renderPaths(state.paths);
      setStatus("train-status", state.training.running, state.training.pid ? `(pid ${state.training.pid})` : "");
      setStatus("eval-status", state.evaluation.running, state.evaluation.pid ? `(pid ${state.evaluation.pid})` : "");
      renderMetrics("train-metrics", state.training.metrics, [
        ["iteration", "iteration_number"],
        ["latest loss", "latest_train_loss"],
        ["best val", "best_val_loss"],
        ["lr", "learning_rate"],
        ["status", "status"],
      ]);
      renderMetrics("eval-metrics", state.evaluation.metrics, [
        ["requested", "requested_tasks"],
        ["submitted", "submitted_tasks"],
        ["completed", "completed_tasks"],
        ["skipped", "skipped_tasks"],
        ["checkpoint", "checkpoint_exists"],
      ]);
      document.getElementById("train-log").textContent = (state.training.log_tail || []).join("\\n");
      document.getElementById("eval-log").textContent = (state.evaluation.log_tail || []).join("\\n");
      renderComparisons(state.evaluation.comparisons);
      document.getElementById("train-start").disabled = state.training.running;
      document.getElementById("train-stop").disabled = !state.training.running;
      document.getElementById("eval-start").disabled = state.evaluation.running || !state.paths.checkpoint_exists;
      document.getElementById("eval-stop").disabled = !state.evaluation.running;
    }

    attachAutoSaveHandlers();
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class MonitorRequestHandler(BaseHTTPRequestHandler):
    app: MonitorApplication = None  # type: ignore

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(as_serializable(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/api/state":
            self._send_json(self.app.snapshot())
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/settings":
                payload = self._read_json_body()
                self.app.update_settings(payload)
                self._send_json({"ok": True})
                return
            if path == "/api/train/start":
                self.app.start_training()
                self._send_json({"ok": True})
                return
            if path == "/api/train/stop":
                self.app.stop_training()
                self._send_json({"ok": True})
                return
            if path == "/api/eval/start":
                self.app.start_evaluation()
                self._send_json({"ok": True})
                return
            if path == "/api/eval/stop":
                self.app.stop_evaluation()
                self._send_json({"ok": True})
                return
            if path == "/api/dialog":
                payload = self._read_json_body()
                field_key = payload.get("field_key", "")
                kind = payload.get("kind", "")
                current_value = payload.get("current_value")
                if "." not in field_key:
                    raise ValueError(f"Invalid field key: {field_key}")
                selected_path = choose_native_path(kind, current_value)
                if not selected_path:
                    self._send_json({"ok": True, "cancelled": True})
                    return
                section, key = field_key.split(".", 1)
                self.app.update_settings({section: {key: selected_path}})
                self._send_json({"ok": True, "path": selected_path})
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local deCIFer training/evaluation monitor.")
    parser.add_argument("--train-config", type=str, default=None, help="Optional training YAML used only to seed initial settings.")
    parser.add_argument("--eval-config", type=str, default=None, help="Optional evaluation YAML used only to seed initial settings.")
    parser.add_argument("--state-file", type=str, default=None, help="Path to the persisted app state JSON.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--python", dest="python_executable", type=str, default="python")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    app = MonitorApplication(args.train_config, args.eval_config, args.python_executable, state_path=args.state_file)
    MonitorRequestHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), MonitorRequestHandler)
    print(f"Serving deCIFer monitor at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
