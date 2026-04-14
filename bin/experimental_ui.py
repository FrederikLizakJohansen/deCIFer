#!/usr/bin/env python3

import __main__
import hashlib
import io
import json
import os
import pickle
import sys
import threading
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "The experimental UI requires `streamlit`. Install it with `pip install streamlit` "
        "and then run `streamlit run bin/experimental_ui.py`."
    ) from exc

from bin.train import TrainConfig
if not hasattr(__main__, "TrainConfig"):
    __main__.TrainConfig = TrainConfig

from bin.experimental_pipeline import DeciferPipeline
from decifer.utility import generate_continuous_xrd_from_cif

CRYSTAL_SYSTEM_OPTIONS = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

STATE_FILE = os.path.join(os.path.expanduser("~"), ".decifer_experimental_ui_state.json")
SESSION_DIR = os.path.join(os.path.expanduser("~"), ".decifer_experimental_ui_sessions")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")

ACCENT = "#6366f1"
ACCENT_SOFT = "#a5b4fc"
EXP_COLOR = "#0ea5e9"
GEN_COLOR = "#f43f5e"
MUTED = "#64748b"


# ---------------------------------------------------------------------------
# State / persistence helpers
# ---------------------------------------------------------------------------

@dataclass
class GenerationJobState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None
    running: bool = False
    finished: bool = False
    stop_all: bool = False
    abort_current: bool = False
    current_generation: int = 0
    total_generations: int = 0
    # Active run's generations (alias into groups[-1]["generations"] when a run is active).
    completed_generations: List[Dict[str, Any]] = field(default_factory=list)
    # All runs performed in this session; each group has its own generations list.
    groups: List[Dict[str, Any]] = field(default_factory=list)
    active_group_index: Optional[int] = None
    latest_partial_text: str = ""
    latest_token_text: str = ""
    token_step: int = 0
    token_total: int = 0
    status_text: str = "Idle"
    last_generation_status: str = ""
    latest_valid_cif: str = ""
    error: Optional[str] = None


def load_persisted_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_persisted_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def list_saved_sessions() -> List[str]:
    if not os.path.isdir(SESSION_DIR):
        return []
    return sorted(
        [name for name in os.listdir(SESSION_DIR) if name.endswith(".pkl")],
        reverse=True,
    )


def build_session_payload(
    pipeline: DeciferPipeline,
    job: GenerationJobState,
    uploaded_name: str,
    uploaded_bytes: bytes,
) -> Dict[str, Any]:
    def _slim_gen(gen: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": gen.get("success", True),
            "status": gen.get("status", "completed"),
            "cif_str": gen.get("cif_str", ""),
            "summary": gen.get("summary", {}),
        }

    with job.lock:
        groups_payload = [
            {
                "id": g.get("id"),
                "name": g.get("name"),
                "status": g.get("status", "done"),
                "created_at": g.get("created_at"),
                "completed_at": g.get("completed_at"),
                "target_total": g.get("target_total", 0),
                "settings": dict(g.get("settings", {})),
                "generations": [_slim_gen(gn) for gn in g.get("generations", [])],
            }
            for g in job.groups
        ]
        # Keep flat list for backwards compatibility with older session files.
        completed_generations = [_slim_gen(gen) for gen in job.completed_generations]
        latest_valid_cif = job.latest_valid_cif

    return {
        "uploaded_name": uploaded_name,
        "uploaded_bytes": uploaded_bytes,
        "settings": {
            "checkpoint_path": st.session_state.get("checkpoint_path", "ckpt.pt"),
            "device": st.session_state.get("device_select", "cpu"),
            "max_new_tokens": st.session_state.get("max_new_tokens_input", 3000),
            "temperature": st.session_state.get("temperature_input", 1.0),
            "composition": st.session_state.get("composition_input", ""),
            "spacegroup": st.session_state.get("spacegroup_input", ""),
            "crystal_systems": st.session_state.get("crystal_systems_input", []),
            "n_generations": st.session_state.get("n_generations_input", 8),
            "wavelength": st.session_state.get("wavelength_input", ""),
            "q_min": st.session_state.get("q_min_input", 1.0),
            "q_max": st.session_state.get("q_max_input", 8.0),
            "interpolation_points": st.session_state.get("interpolation_points_input", 1000),
        },
        "preview": {
            "df_exp": pipeline.df_exp.to_dict(orient="list"),
            "df_processed": pipeline.df_processed.to_dict(orient="list") if pipeline.df_processed is not None else None,
            "exp_q": pipeline.exp_q.tolist() if pipeline.exp_q is not None else None,
            "exp_i": pipeline.exp_i.tolist() if pipeline.exp_i is not None else None,
            "model_condition_size": pipeline.model_condition_size,
        },
        "generations": completed_generations,
        "groups": groups_payload,
        "latest_valid_cif": latest_valid_cif,
    }


def save_named_session(session_name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(SESSION_DIR, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_name).strip("_") or "session"
    filename = f"{safe_name}.pkl"
    path = os.path.join(SESSION_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return filename


def load_saved_session(filename: str) -> Dict[str, Any]:
    path = os.path.join(SESSION_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_saved_session(filename: str) -> bool:
    path = os.path.join(SESSION_DIR, filename)
    try:
        os.remove(path)
        return True
    except Exception:
        return False


def apply_loaded_session(payload: Dict[str, Any], job: GenerationJobState) -> None:
    """Apply a loaded session payload to session_state and job. Must be called BEFORE widgets render."""
    settings = payload.get("settings", {})
    groups = payload.get("groups")
    legacy_flat = payload.get("generations", [])
    if not groups and legacy_flat:
        # Wrap a legacy flat generations list into a single group.
        groups = [{
            "id": "run-legacy",
            "name": "Run 1 (imported)",
            "status": "done",
            "created_at": "",
            "completed_at": "",
            "target_total": settings.get("n_generations", len(legacy_flat)),
            "settings": {
                "composition": settings.get("composition", ""),
                "spacegroup": settings.get("spacegroup", ""),
                "crystal_systems": settings.get("crystal_systems", []),
                "temperature": settings.get("temperature", 1.0),
                "max_new_tokens": settings.get("max_new_tokens", 3000),
                "n_generations": settings.get("n_generations", len(legacy_flat)),
            },
            "generations": list(legacy_flat),
        }]
    groups = groups or []

    with job.lock:
        job.running = False
        job.finished = True
        job.stop_all = False
        job.abort_current = False
        job.groups = [dict(g) for g in groups]
        job.active_group_index = (len(job.groups) - 1) if job.groups else None
        active_gens = job.groups[-1]["generations"] if job.groups else []
        job.current_generation = len(active_gens)
        job.total_generations = settings.get("n_generations", len(active_gens))
        job.completed_generations = active_gens
        job.latest_partial_text = ""
        job.latest_token_text = ""
        job.token_step = 0
        job.token_total = settings.get("max_new_tokens", 3000)
        job.status_text = "Loaded saved session"
        job.last_generation_status = ""
        job.latest_valid_cif = payload.get("latest_valid_cif", "")
        job.error = None
    st.session_state.loaded_session_payload = payload
    st.session_state.pending_checkpoint_path = settings.get("checkpoint_path", "ckpt.pt")
    # These keys are set BEFORE widgets are created on next rerun, so assignment is safe.
    for key, value in {
        "device_select": settings.get("device", "cpu"),
        "max_new_tokens_input": settings.get("max_new_tokens", 3000),
        "temperature_input": settings.get("temperature", 1.0),
        "composition_input": settings.get("composition", ""),
        "spacegroup_input": settings.get("spacegroup", ""),
        "crystal_systems_input": settings.get("crystal_systems", []),
        "n_generations_input": settings.get("n_generations", 8),
        "wavelength_input": settings.get("wavelength", ""),
        "q_min_input": settings.get("q_min", 1.0),
        "q_max_input": settings.get("q_max", 8.0),
        "interpolation_points_input": settings.get("interpolation_points", 1000),
    }.items():
        st.session_state.pop(key, None)
        st.session_state[key] = value


def make_cif_zip(generations: List[Dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, gen in enumerate(generations, start=1):
            summary = gen.get("summary", {})
            formula = summary.get("formula") or "unknown"
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(formula))
            zf.writestr(f"generation_{idx:02d}_{safe}.cif", gen.get("cif_str", ""))
    return buf.getvalue()


def make_overlay_figure(exp_q, exp_i, generations: List[Dict[str, Any]], q_min: float, q_max: float) -> Optional[go.Figure]:
    exp_q = np.asarray(exp_q)
    exp_i = np.asarray(exp_i)
    mask = (exp_q >= q_min) & (exp_q <= q_max)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exp_q[mask], y=exp_i[mask], mode="lines", name="Experimental",
            line=dict(color=EXP_COLOR, width=2.4),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
        )
    )
    palette = ["#f43f5e", "#f97316", "#eab308", "#84cc16", "#10b981", "#06b6d4", "#8b5cf6", "#ec4899"]
    for idx, gen in enumerate(generations):
        pxrd = _compute_pxrd_cached(gen.get("cif_str", ""), float(q_min), float(q_max))
        if pxrd is None:
            continue
        disc_mask = (pxrd["q_disc"] >= q_min) & (pxrd["q_disc"] <= q_max)
        if not disc_mask.any():
            continue
        disc_i = pxrd["iq_disc"][disc_mask] / max(pxrd["iq_disc"][disc_mask].max(), 1e-12)
        qs = pxrd["q_disc"][disc_mask]
        color = palette[idx % len(palette)]
        label = f"#{idx+1} {gen.get('summary', {}).get('formula', '?')}"
        xs, ys = [], []
        offset = -0.04 * (idx + 1)
        for q_val, i_val in zip(qs, disc_i):
            xs.extend([q_val, q_val, None])
            ys.extend([offset, offset - i_val * 0.15, None])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="name"))
    fig.update_xaxes(title_text="Q (Å⁻¹)", range=[q_min, q_max])
    fig.update_yaxes(title_text="Intensity (stacked tick marks below)")
    return _apply_layout(fig, height=420, title="Overlay: experimental vs all valid structures")


def get_job_state() -> GenerationJobState:
    if "generation_job" not in st.session_state:
        st.session_state.generation_job = GenerationJobState()
    return st.session_state.generation_job


def sync_checkpoint_widget_state() -> None:
    persisted = load_persisted_state()
    if "pending_checkpoint_path" in st.session_state:
        pending_path = st.session_state.pop("pending_checkpoint_path")
        st.session_state.checkpoint_path = pending_path
        st.session_state.checkpoint_path_input = pending_path
    if "checkpoint_path" not in st.session_state:
        st.session_state.checkpoint_path = persisted.get("checkpoint_path", "ckpt.pt")
    if "checkpoint_path_input" not in st.session_state:
        st.session_state.checkpoint_path_input = st.session_state.get("checkpoint_path", "ckpt.pt")


def save_ui_state() -> None:
    save_persisted_state(
        {
            "checkpoint_path": st.session_state.get("checkpoint_path", "ckpt.pt"),
            "device": st.session_state.get("device_select", "cpu"),
            "max_new_tokens": st.session_state.get("max_new_tokens_input", 3000),
            "temperature": st.session_state.get("temperature_input", 1.0),
            "composition": st.session_state.get("composition_input", ""),
            "spacegroup": st.session_state.get("spacegroup_input", ""),
            "crystal_systems": st.session_state.get("crystal_systems_input", []),
            "n_generations": st.session_state.get("n_generations_input", 8),
            "wavelength": st.session_state.get("wavelength_input", ""),
            "q_min": st.session_state.get("q_min_input", 1.0),
            "q_max": st.session_state.get("q_max_input", 8.0),
            "interpolation_points": st.session_state.get("interpolation_points_input", 1000),
        }
    )


def choose_checkpoint_file() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Choose deCIFer checkpoint",
            filetypes=[
                ("PyTorch checkpoint", "*.pt *.pth *.ckpt"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return selected or None
    except Exception:
        return None


def compute_preview_key(
    checkpoint_path: str,
    device: str,
    uploaded_name: str,
    uploaded_bytes: bytes,
    q_min: float,
    q_max: float,
    interpolation_points: int,
    wavelength_text: str,
) -> str:
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    return "|".join(
        [
            checkpoint_path,
            device,
            uploaded_name,
            digest,
            str(q_min),
            str(q_max),
            str(interpolation_points),
            wavelength_text.strip(),
        ]
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_layout(fig: go.Figure, height: int = 340, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title or "", x=0.0, xanchor="left", font=dict(size=14, color="#0f172a")),
        height=height,
        margin=dict(l=50, r=20, t=50 if title else 20, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", color="#0f172a"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#e2e8f0", zeroline=False,
            showline=True, linecolor="#cbd5e1", ticks="outside", tickcolor="#cbd5e1",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e2e8f0", zeroline=False,
            showline=True, linecolor="#cbd5e1", ticks="outside", tickcolor="#cbd5e1",
        ),
    )
    return fig


def make_signal_figure(df_processed: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_processed["Q"],
            y=df_processed["intensity_normalized"],
            mode="lines",
            name="Raw (normalized)",
            line=dict(color=MUTED, width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_processed["Q"],
            y=df_processed["intensity_crop_normalized"],
            mode="lines",
            name="Model input (cropped & interpolated)",
            line=dict(color=EXP_COLOR, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.08)",
        )
    )
    fig.update_xaxes(title_text="Q (Å⁻¹)")
    fig.update_yaxes(title_text="Normalized intensity")
    return _apply_layout(fig, height=360)


@st.cache_data(show_spinner=False, max_entries=512)
def _compute_pxrd_cached(cif_str: str, q_min: float, q_max: float) -> Optional[Dict[str, Any]]:
    try:
        return generate_continuous_xrd_from_cif(
            cif_str,
            qmin=q_min, qmax=q_max, qstep=0.01,
            fwhm_range=(0.05, 0.05), eta_range=(0.5, 0.5),
            noise_range=None, intensity_scale_range=None, mask_prob=None,
        )
    except Exception:
        return None


def make_generation_figure(exp_q, exp_i, cif_str: str, q_min: float, q_max: float) -> Optional[go.Figure]:
    pxrd = _compute_pxrd_cached(cif_str, float(q_min), float(q_max))
    if pxrd is None:
        return None

    exp_mask = (exp_q >= q_min) & (exp_q <= q_max)
    disc_mask = (pxrd["q_disc"] >= q_min) & (pxrd["q_disc"] <= q_max)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(exp_q)[exp_mask],
            y=np.asarray(exp_i)[exp_mask],
            mode="lines",
            name="Experimental",
            line=dict(color=EXP_COLOR, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.08)",
        )
    )
    if disc_mask.any():
        disc_i = pxrd["iq_disc"][disc_mask] / max(pxrd["iq_disc"][disc_mask].max(), 1e-12)
        qs = pxrd["q_disc"][disc_mask]
        xs, ys = [], []
        for q_val, i_val in zip(qs, disc_i):
            xs.extend([q_val, q_val, None])
            ys.extend([0.0, i_val, None])
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=GEN_COLOR, width=1.4),
                name="Generated reflections",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=qs, y=disc_i, mode="markers",
                marker=dict(color=GEN_COLOR, size=6, symbol="circle", line=dict(color="white", width=1)),
                name="Peak",
                showlegend=False,
            )
        )
    fig.update_xaxes(title_text="Q (Å⁻¹)", range=[q_min, q_max])
    fig.update_yaxes(title_text="Normalized intensity")
    return _apply_layout(fig, height=320)


def generation_rows(gens: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, gen in enumerate(gens, start=1):
        summary = gen.get("summary", {})
        rows.append(
            {
                "#": idx,
                "formula": summary.get("formula"),
                "spacegroup": summary.get("spacegroup"),
                "crystal system": summary.get("crystal_system"),
                "atoms": summary.get("n_atoms"),
                "status": gen.get("status", "completed"),
            }
        )
    return pd.DataFrame(rows)


def generation_labels(gens: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for idx, gen in enumerate(gens, start=1):
        summary = gen.get("summary", {})
        formula = summary.get("formula") or "unknown"
        spacegroup = summary.get("spacegroup") or "—"
        labels.append(f"#{idx}  ·  {formula}  ·  {spacegroup}")
    return labels


def prepare_preview_pipeline(
    checkpoint_path: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    uploaded_name: str,
    uploaded_bytes: bytes,
    q_min: float,
    q_max: float,
    interpolation_points: int,
    wavelength_text: str,
) -> DeciferPipeline:
    preview_key = compute_preview_key(
        checkpoint_path=checkpoint_path,
        device=device,
        uploaded_name=uploaded_name,
        uploaded_bytes=uploaded_bytes,
        q_min=q_min,
        q_max=q_max,
        interpolation_points=interpolation_points,
        wavelength_text=wavelength_text,
    )
    if st.session_state.get("preview_key") == preview_key and "preview_pipeline" in st.session_state:
        return st.session_state.preview_pipeline

    wavelength = None if not wavelength_text.strip() else float(wavelength_text)
    pipeline = DeciferPipeline(
        model_path=checkpoint_path,
        device=device,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
    )
    pipeline.load_experimental_file(uploaded_bytes, uploaded_name)
    pipeline.prepare_target_data(
        target_file=uploaded_name,
        q_min_crop=float(q_min),
        q_max_crop=float(q_max),
        wavelength=wavelength,
        n_points=int(interpolation_points),
    )
    st.session_state.preview_key = preview_key
    st.session_state.preview_pipeline = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------

def start_generation_job(
    job: GenerationJobState,
    checkpoint_path: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    uploaded_name: str,
    uploaded_bytes: bytes,
    q_min: float,
    q_max: float,
    interpolation_points: int,
    wavelength_text: str,
    n_generations: int,
    composition: Optional[str],
    spacegroup: Optional[str],
    crystal_systems: Optional[List[str]],
) -> None:
    if job.running:
        return

    with job.lock:
        new_group = {
            "id": f"run-{len(job.groups) + 1}-{int(datetime.now().timestamp())}",
            "name": f"Run {len(job.groups) + 1}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_total": int(n_generations),
            "generations": [],
            "status": "running",
            "settings": {
                "composition": composition,
                "spacegroup": spacegroup,
                "crystal_systems": list(crystal_systems) if crystal_systems else [],
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "n_generations": int(n_generations),
            },
        }
        job.groups.append(new_group)
        job.active_group_index = len(job.groups) - 1
        job.thread = None
        job.running = True
        job.finished = False
        job.stop_all = False
        job.abort_current = False
        job.current_generation = 0
        job.total_generations = int(n_generations)
        job.completed_generations = new_group["generations"]  # alias → appends land in the group
        job.latest_partial_text = ""
        job.latest_token_text = ""
        job.token_step = 0
        job.token_total = int(max_new_tokens)
        job.status_text = f"Preparing {new_group['name']}…"
        job.last_generation_status = ""
        job.latest_valid_cif = ""
        job.error = None

    def worker() -> None:
        try:
            wavelength = None if not wavelength_text.strip() else float(wavelength_text)
            pipeline = DeciferPipeline(
                model_path=checkpoint_path,
                device=device,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
            )
            pipeline.load_experimental_file(uploaded_bytes, uploaded_name)
            pipeline.prepare_target_data(
                target_file=uploaded_name,
                q_min_crop=float(q_min),
                q_max_crop=float(q_max),
                wavelength=wavelength,
                n_points=int(interpolation_points),
            )

            def should_stop_all() -> bool:
                with job.lock:
                    return job.stop_all

            def should_abort_current() -> bool:
                with job.lock:
                    return job.abort_current

            def on_token(current_generation, total_generations, token_step, token_total, token_text, partial_text):
                with job.lock:
                    job.current_generation = current_generation
                    job.total_generations = total_generations
                    job.token_step = token_step
                    job.token_total = token_total
                    job.latest_token_text = token_text
                    job.latest_partial_text = partial_text
                    job.status_text = f"Sampling structure {current_generation}/{total_generations} — token {token_step}/{token_total}"

            def on_generation_done(current: int, total: int, generation_result: Dict[str, Any]) -> None:
                with job.lock:
                    job.current_generation = current
                    job.total_generations = total
                    job.last_generation_status = generation_result.get("status", "")
                    if generation_result.get("success"):
                        job.completed_generations.append(generation_result)
                        job.latest_valid_cif = generation_result["cif_str"]
                        job.status_text = f"✓ Structure {current}/{total} valid"
                    elif generation_result.get("status") == "aborted":
                        job.status_text = f"Structure {current}/{total} abandoned"
                        job.abort_current = False
                    elif generation_result.get("status") == "stopped":
                        job.status_text = "Stopped"
                    else:
                        job.status_text = f"Structure {current}/{total} invalid — retrying"

            pipeline.run_experiment_protocol(
                n_trials=int(n_generations),
                composition=composition,
                spacegroup=spacegroup,
                crystal_systems=crystal_systems or None,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                protocol_name=uploaded_name,
                progress_callback=on_generation_done,
                token_callback=on_token,
                should_stop_all_callback=should_stop_all,
                should_abort_current_callback=should_abort_current,
            )
        except Exception:
            with job.lock:
                job.error = traceback.format_exc()
                job.status_text = "Generation failed"
                if job.active_group_index is not None and job.active_group_index < len(job.groups):
                    job.groups[job.active_group_index]["status"] = "failed"
        finally:
            with job.lock:
                job.running = False
                job.finished = True
                if job.active_group_index is not None and job.active_group_index < len(job.groups):
                    grp = job.groups[job.active_group_index]
                    if grp.get("status") == "running":
                        grp["status"] = "done"
                    grp["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    thread = threading.Thread(target=worker, daemon=True)
    with job.lock:
        job.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Live status panel
# ---------------------------------------------------------------------------

def _group_label(idx: int, group: Dict[str, Any]) -> str:
    status = group.get("status", "done")
    valid = len(group.get("generations", []))
    target = group.get("target_total", 0) or "?"
    icon = {"running": "●", "done": "✓", "failed": "✗"}.get(status, "•")
    settings = group.get("settings", {})
    bits = []
    if settings.get("composition"): bits.append(settings["composition"])
    if settings.get("spacegroup"): bits.append(f"SG {settings['spacegroup']}")
    tag = " · ".join(bits) if bits else "no constraints"
    return f"{icon} {group.get('name', f'Run {idx+1}')}  ·  {valid}/{target}  ·  {tag}"


@st.fragment(run_every="750ms")
def render_job_status(job: GenerationJobState, exp_q, exp_i, q_min: float, q_max: float) -> None:
    with job.lock:
        running = job.running
        finished = job.finished
        stop_all = job.stop_all
        abort_current = job.abort_current
        current_generation = job.current_generation
        total_generations = job.total_generations
        token_step = job.token_step
        token_total = job.token_total
        status_text = job.status_text
        latest_partial_text = job.latest_partial_text
        error = job.error
        active_idx = job.active_group_index
        # Deep-copy minimal group shape for safe read outside lock.
        groups_snapshot = [
            {
                "id": g.get("id"),
                "name": g.get("name"),
                "status": g.get("status"),
                "created_at": g.get("created_at"),
                "completed_at": g.get("completed_at"),
                "target_total": g.get("target_total", 0),
                "settings": dict(g.get("settings", {})),
                "generations": list(g.get("generations", [])),
            }
            for g in job.groups
        ]

    # Handle a pending group deletion from the previous render of this fragment.
    pending_del = st.session_state.pop("pending_group_delete", None)
    if pending_del is not None:
        with job.lock:
            for i, g in enumerate(job.groups):
                if g.get("id") == pending_del and g.get("status") != "running":
                    del job.groups[i]
                    if job.active_group_index is not None:
                        if job.active_group_index == i:
                            job.active_group_index = None
                        elif job.active_group_index > i:
                            job.active_group_index -= 1
                    break
        st.rerun()

    total_valid = sum(len(g["generations"]) for g in groups_snapshot)

    # Top-level metrics
    state_label = "● Running" if running else ("✓ Finished" if finished else "○ Idle")
    state_color = "#059669" if running else ("#6366f1" if finished else MUTED)
    m = st.columns(4)
    m[0].metric("Total runs", len(groups_snapshot))
    m[1].metric("Total valid structures", total_valid)
    if running and total_generations:
        m[2].metric("Current run", f"{current_generation}/{total_generations}")
    else:
        m[2].metric("Current run", "—")
    m[3].markdown(
        f"<div style='font-size:0.85rem;color:{MUTED};margin-top:4px'>State</div>"
        f"<div style='font-weight:600;color:{state_color};font-size:1.4rem'>{state_label}</div>",
        unsafe_allow_html=True,
    )

    # Active run progress + controls
    if running and total_generations:
        active_valid = len(groups_snapshot[active_idx]["generations"]) if active_idx is not None and active_idx < len(groups_snapshot) else 0
        overall = min((active_valid + 1) / total_generations, 1.0) if total_generations else 0.0
        st.progress(overall, text=status_text)
        if token_total:
            st.progress(min(token_step / token_total, 1.0), text=f"Token {token_step}/{token_total}")
        c = st.columns(2)
        if c[0].button("⏭ Skip current", disabled=abort_current, width="stretch", key="abort_btn"):
            with job.lock:
                job.abort_current = True
                job.status_text = "Abandoning current structure — takes effect at the next token."
        if c[1].button("⏹ Stop all", disabled=stop_all, width="stretch", type="primary", key="stop_btn"):
            with job.lock:
                job.stop_all = True
                job.status_text = "Stopping — takes effect at the next token."

    if error:
        st.error("Generation job failed.")
        with st.expander("Traceback"):
            st.code(error, language="text")

    if not groups_snapshot:
        st.info("No runs yet — click **Run generation** above to start.")
        return

    st.markdown("##### Runs")

    # Run picker
    group_options = [_group_label(i, g) for i, g in enumerate(groups_snapshot)]
    default_run_idx = active_idx if (active_idx is not None and active_idx < len(group_options)) else len(group_options) - 1
    sel_key = "active_run_select"
    if sel_key not in st.session_state or st.session_state[sel_key] not in group_options:
        st.session_state[sel_key] = group_options[default_run_idx]
    # If a run just finished, jump to the latest. Otherwise respect the user's selection.
    if running and group_options[default_run_idx] != st.session_state[sel_key] and active_idx is not None:
        st.session_state[sel_key] = group_options[active_idx]
    selected_run_label = st.radio(
        "run_picker", options=group_options, key=sel_key,
        label_visibility="collapsed", horizontal=False,
    )
    selected_run_idx = group_options.index(selected_run_label)
    group = groups_snapshot[selected_run_idx]
    group_gens = group["generations"]
    is_running_group = running and active_idx == selected_run_idx

    # Run-level actions
    run_action_cols = st.columns([1, 1, 1, 2])
    run_action_cols[0].download_button(
        "⬇ CIFs (.zip)",
        data=make_cif_zip(group_gens) if group_gens else b"",
        file_name=f"{group['name'].replace(' ', '_')}_cifs.zip",
        mime="application/zip",
        width="stretch",
        disabled=not group_gens,
        key=f"dl_zip_{group['id']}",
    )
    if group_gens:
        summary_csv = generation_rows(group_gens).to_csv(index=False).encode("utf-8")
    else:
        summary_csv = b""
    run_action_cols[1].download_button(
        "⬇ Summary (.csv)", data=summary_csv,
        file_name=f"{group['name'].replace(' ', '_')}_summary.csv",
        mime="text/csv", width="stretch",
        disabled=not group_gens, key=f"dl_csv_{group['id']}",
    )
    if run_action_cols[2].button(
        "🗑 Delete run", width="stretch",
        disabled=is_running_group,
        help="Stop the run before deleting it." if is_running_group else None,
        key=f"del_{group['id']}",
    ):
        st.session_state.pending_group_delete = group["id"]
        st.rerun()
    run_action_cols[3].markdown(
        f"<div style='color:{MUTED};font-size:0.85rem;padding-top:10px'>"
        f"Started {group.get('created_at', '—')}"
        + (f" · finished {group.get('completed_at')}" if group.get("completed_at") else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Structure picker within selected run
    if group_gens:
        labels = generation_labels(group_gens)
        struct_key = f"struct_pick_{group['id']}"
        if struct_key not in st.session_state or st.session_state[struct_key] not in labels:
            st.session_state[struct_key] = labels[-1]
        selected_label = st.radio(
            "structure_picker", options=labels, key=struct_key,
            label_visibility="collapsed", horizontal=False,
        )
        sidx = labels.index(selected_label)
        selected = group_gens[sidx]

        detail_cols = st.columns([1.3, 1])
        with detail_cols[0]:
            fig = make_generation_figure(exp_q, exp_i, selected["cif_str"], q_min=q_min, q_max=q_max)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Could not render diffraction for this structure.")
        with detail_cols[1]:
            summary = selected.get("summary", {})
            st.markdown("**Summary**")
            items = [
                ("Formula", summary.get("formula") or "—"),
                ("Space group", summary.get("spacegroup") or "—"),
                ("Crystal system", summary.get("crystal_system") or "—"),
                ("Atoms", summary.get("n_atoms") or "—"),
            ]
            html = "<table style='width:100%;font-size:0.9rem'>"
            for k, v in items:
                html += f"<tr><td style='color:{MUTED};padding:4px 0'>{k}</td><td style='text-align:right;font-weight:500'>{v}</td></tr>"
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)
            st.download_button(
                "⬇ Download CIF",
                data=selected["cif_str"],
                file_name=f"{group['name'].replace(' ', '_')}_gen_{sidx + 1}.cif",
                mime="text/plain", width="stretch",
                key=f"dl_{group['id']}_{sidx + 1}",
            )

        with st.expander("CIF text"):
            st.code(selected["cif_str"], language="text")
        with st.expander("Run table"):
            st.dataframe(generation_rows(group_gens), width="stretch", hide_index=True)
    elif is_running_group:
        st.info("Waiting for the first valid structure in this run…")
        if latest_partial_text:
            with st.expander("Live token stream"):
                st.code(latest_partial_text[-2000:], language="text")
    else:
        st.warning("This run produced no valid structures.")

    # Cross-run overlay
    st.markdown("##### Compare across runs")
    if len(groups_snapshot) >= 1:
        run_names = [g["name"] for g in groups_snapshot if g["generations"]]
        if run_names:
            chosen_runs = st.multiselect(
                "Include runs in overlay",
                options=run_names,
                default=run_names,
                key="overlay_runs_select",
            )
            show_overlay = st.toggle("Show overlay", value=False, key="overlay_toggle_global")
            if show_overlay and chosen_runs:
                combined: List[Dict[str, Any]] = []
                for g in groups_snapshot:
                    if g["name"] in chosen_runs:
                        for gen in g["generations"]:
                            tagged = dict(gen)
                            s = dict(tagged.get("summary", {}))
                            s["formula"] = f"[{g['name']}] {s.get('formula', '?')}"
                            tagged["summary"] = s
                            combined.append(tagged)
                if combined:
                    overlay_fig = make_overlay_figure(exp_q, exp_i, combined, q_min=q_min, q_max=q_max)
                    if overlay_fig is not None:
                        st.plotly_chart(overlay_fig, width="stretch")
        else:
            st.caption("No valid structures to overlay yet.")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* base typography */
html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

/* hero */
.hero-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: #0f172a; }
.hero-sub { color: #64748b; font-size: 1rem; margin-top: 4px; margin-bottom: 1.5rem; }

/* section cards */
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.section-title {
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6366f1; margin: 0 0 12px 0;
}
.section-h { font-size: 1.1rem; font-weight: 600; color: #0f172a; margin: 0 0 6px 0; }
.section-desc { color: #64748b; font-size: 0.88rem; margin-bottom: 14px; }

/* ready pill */
.pill { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-size:0.78rem; font-weight:500; }
.pill-ok { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.pill-warn { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.pill-err { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }

/* primary CTA */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    font-weight: 600;
    padding: 0.6rem 1rem;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    transform: translateY(-1px);
}

/* sidebar */
section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; }

/* metric */
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
[data-testid="stMetricLabel"] { color: #64748b; font-size: 0.8rem; }

/* file uploader */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #cbd5e1;
    background: #f8fafc;
    border-radius: 10px;
    transition: all 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366f1;
    background: #eef2ff;
}

/* radio (gallery picker) */
div[role="radiogroup"] label {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    background: #ffffff;
    transition: all 0.15s ease;
}
div[role="radiogroup"] label:hover { border-color: #a5b4fc; background: #f5f3ff; }
div[role="radiogroup"] label[data-checked="true"] { border-color: #6366f1; background: #eef2ff; }
</style>
"""


def pill(text: str, kind: str = "ok") -> str:
    return f"<span class='pill pill-{kind}'>{text}</span>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="deCIFer — Experimental UI",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    persisted = load_persisted_state()
    if "checkpoint_path" not in st.session_state:
        st.session_state.checkpoint_path = persisted.get("checkpoint_path", "ckpt.pt")
    sync_checkpoint_widget_state()
    job = get_job_state()

    # Process any pending session load BEFORE widgets render (avoids StreamlitAPIException).
    pending_load = st.session_state.pop("pending_session_load", None)
    if pending_load:
        try:
            payload = load_saved_session(pending_load)
            apply_loaded_session(payload, job)
            sync_checkpoint_widget_state()
            persisted = load_persisted_state()
            st.toast(f"Loaded session: {pending_load}", icon="📂")
        except Exception as exc:
            st.error(f"Failed to load session: {exc}")

    pending_delete = st.session_state.pop("pending_session_delete", None)
    if pending_delete:
        if delete_saved_session(pending_delete):
            st.toast(f"Deleted: {pending_delete}", icon="🗑")
        else:
            st.toast(f"Could not delete: {pending_delete}", icon="⚠")

    # ---------- Sidebar: model + session management ----------
    with st.sidebar:
        st.markdown("### 🔬 deCIFer")
        st.caption("Experimental diffraction → crystal structure")

        st.markdown("---")
        st.markdown("#### Model")
        cp_path = st.text_input(
            "Checkpoint path",
            value=st.session_state.checkpoint_path,
            key="checkpoint_path_input",
            label_visibility="collapsed",
            placeholder="path/to/ckpt.pt",
        )
        st.session_state.checkpoint_path = cp_path
        if st.button("📁 Browse…", key="select_checkpoint_file_button"):
            selected_path = choose_checkpoint_file()
            if selected_path:
                st.session_state.pending_checkpoint_path = selected_path
                st.session_state.checkpoint_path = selected_path
                save_ui_state()
                st.rerun()

        cp_ok = os.path.exists(st.session_state.checkpoint_path)
        st.markdown(
            pill("Checkpoint ready", "ok") if cp_ok else pill("Checkpoint missing", "err"),
            unsafe_allow_html=True,
        )

        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        persisted_device = persisted.get("device", device_options[0])
        device_index = device_options.index(persisted_device) if persisted_device in device_options else 0
        device = st.selectbox("Device", options=device_options, index=device_index, key="device_select")

        st.markdown("---")
        st.markdown("#### Sessions")
        saved_sessions = list_saved_sessions()
        selected_session = st.selectbox(
            "Load a saved session",
            options=["—"] + saved_sessions,
            index=0,
            key="saved_session_select",
        )
        sess_cols = st.columns(2)
        if sess_cols[0].button("📂 Load", disabled=(selected_session == "—"), key="load_selected_session_button"):
            st.session_state.pending_session_load = selected_session
            st.rerun()
        if sess_cols[1].button("🗑 Delete", disabled=(selected_session == "—"), key="delete_session_btn"):
            st.session_state.pending_session_delete = selected_session
            st.rerun()

        session_name = st.text_input("Session name", value=st.session_state.get("session_name_input", "decifer_session"), key="session_name_input")
        save_clicked = st.button("💾 Save current", disabled=job.running, key="save_session_btn")

        st.markdown("---")
        total_valid = sum(len(g.get("generations", [])) for g in job.groups)
        if job.running:
            st.markdown(pill(f"● Running — {len(job.groups)} run(s)", "warn"), unsafe_allow_html=True)
        elif job.groups:
            st.markdown(pill(f"✓ {len(job.groups)} run(s), {total_valid} structures", "ok"), unsafe_allow_html=True)

    checkpoint_path = st.session_state.checkpoint_path

    # ---------- Header ----------
    st.markdown("<h1 class='hero-title'>Experimental Diffraction Studio</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>Upload a pattern, tune preprocessing, constrain chemistry, and sample structures.</p>",
        unsafe_allow_html=True,
    )

    loaded_session = st.session_state.get("loaded_session_payload")

    # ---------- Step 1: Data ----------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Step 1 · Data</div>", unsafe_allow_html=True)
    data_cols = st.columns([1.1, 1])
    with data_cols[0]:
        st.markdown("<div class='section-h'>Experimental pattern</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-desc'>Drop a <code>.xy</code> or <code>.xye</code> file. Two columns: 2θ (or Q) and intensity.</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload pattern", type=["xy", "xye"], label_visibility="collapsed",
        )
    with data_cols[1]:
        st.markdown("<div class='section-h'>Signal preparation</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-desc'>If your x-axis is 2θ, provide a wavelength to convert to Q.</div>",
            unsafe_allow_html=True,
        )
        wavelength_text = st.text_input(
            "Wavelength (Å) — leave blank if x is already Q",
            value=persisted.get("wavelength", ""),
            key="wavelength_input",
            placeholder="e.g. 1.5406",
        )
        q_cols = st.columns(2)
        q_min = q_cols[0].number_input(
            "Q-min (Å⁻¹)", min_value=0.0, max_value=20.0,
            value=float(persisted.get("q_min", 1.0)), step=0.1, key="q_min_input",
        )
        q_max = q_cols[1].number_input(
            "Q-max (Å⁻¹)", min_value=0.1, max_value=20.0,
            value=float(persisted.get("q_max", 8.0)), step=0.1, key="q_max_input",
        )
        with st.expander("Advanced"):
            interpolation_points = st.number_input(
                "Interpolation points", min_value=64, max_value=4096,
                value=int(persisted.get("interpolation_points", 1000)),
                step=64, key="interpolation_points_input",
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # Resolve data source
    if uploaded_file is not None:
        uploaded_name = uploaded_file.name
        uploaded_bytes = uploaded_file.getvalue()
    elif loaded_session is not None:
        uploaded_name = loaded_session["uploaded_name"]
        uploaded_bytes = loaded_session["uploaded_bytes"]
    else:
        st.info("⬆ Upload an `.xy` / `.xye` file above, or load a saved session from the sidebar.")
        save_ui_state()
        return

    # Validate checkpoint
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint not found: `{checkpoint_path}` — set a valid path in the sidebar.")
        save_ui_state()
        return

    # Build pipeline preview (uses placeholder values for gen params still needed)
    tmp_max_tokens = int(persisted.get("max_new_tokens", 3000))
    tmp_temp = float(persisted.get("temperature", 1.0))
    try:
        pipeline = prepare_preview_pipeline(
            checkpoint_path=checkpoint_path,
            device=device,
            max_new_tokens=tmp_max_tokens,
            temperature=tmp_temp,
            uploaded_name=uploaded_name,
            uploaded_bytes=uploaded_bytes,
            q_min=float(q_min),
            q_max=float(q_max),
            interpolation_points=int(interpolation_points),
            wavelength_text=wavelength_text,
        )
    except Exception as exc:
        st.error("Failed to prepare pattern.")
        st.exception(exc)
        return

    # ---------- Preview ----------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Preview</div>", unsafe_allow_html=True)
    prev_cols = st.columns([2.2, 1])
    with prev_cols[0]:
        st.plotly_chart(make_signal_figure(pipeline.df_processed), width="stretch")
    with prev_cols[1]:
        st.markdown("<div class='section-h'>Readiness</div>", unsafe_allow_html=True)
        rc = st.columns(2)
        rc[0].metric("Raw points", int(len(pipeline.df_exp)))
        rc[1].metric("Model input", int(interpolation_points))
        st.markdown(f"**File**  <span style='color:{MUTED}'>{uploaded_name}</span>", unsafe_allow_html=True)
        st.markdown(f"**Q window**  {float(q_min):.2f} – {float(q_max):.2f} Å⁻¹")
        if pipeline.model_condition_size and int(interpolation_points) != pipeline.model_condition_size:
            st.warning(f"Model expects {pipeline.model_condition_size} points — will resample.")
        else:
            st.markdown(pill("Matches model input size", "ok"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Step 2: Constraints + sampling ----------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Step 2 · Constraints &amp; sampling</div>", unsafe_allow_html=True)

    constr_cols = st.columns([1, 1, 1])
    with constr_cols[0]:
        st.markdown("<div class='section-h'>Composition</div>", unsafe_allow_html=True)
        composition = st.text_input(
            "Composition", value=persisted.get("composition", ""),
            key="composition_input", label_visibility="collapsed",
            placeholder="e.g. BaTiO3 (optional)",
        )
        st.caption("Chemical formula. Leave blank to let the model decide.")
    with constr_cols[1]:
        st.markdown("<div class='section-h'>Crystal systems</div>", unsafe_allow_html=True)
        crystal_systems = st.multiselect(
            "Crystal systems", options=CRYSTAL_SYSTEM_OPTIONS,
            default=[cs for cs in persisted.get("crystal_systems", []) if cs in CRYSTAL_SYSTEM_OPTIONS],
            key="crystal_systems_input", label_visibility="collapsed",
            placeholder="Any",
        )
        st.caption("Restrict to one or more systems (optional).")
    with constr_cols[2]:
        st.markdown("<div class='section-h'>Space group</div>", unsafe_allow_html=True)
        spacegroup = st.text_input(
            "Space group", value=persisted.get("spacegroup", ""),
            key="spacegroup_input", label_visibility="collapsed",
            placeholder="e.g. Pm-3m (requires composition)",
        )
        st.caption("Exact space group symbol. Requires composition.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    samp_cols = st.columns(3)
    n_generations = samp_cols[0].number_input(
        "Number of generations", min_value=1, max_value=100,
        value=int(persisted.get("n_generations", 8)), step=1, key="n_generations_input",
    )
    temperature = samp_cols[1].number_input(
        "Temperature", min_value=0.1, max_value=2.0,
        value=float(persisted.get("temperature", 1.0)), step=0.1, key="temperature_input",
        help="Higher = more diverse, lower = more conservative.",
    )
    max_new_tokens = samp_cols[2].number_input(
        "Max new tokens", min_value=64, max_value=8000,
        value=int(persisted.get("max_new_tokens", 3000)), step=64, key="max_new_tokens_input",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    save_ui_state()

    # Validation
    if spacegroup.strip() and not composition.strip():
        st.error("❌ Composition is required when a space group is specified.")
        return

    # ---------- Step 3: Run ----------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Step 3 · Generate</div>", unsafe_allow_html=True)

    run_cols = st.columns([1, 2])
    with run_cols[0]:
        run_generation = st.button(
            "▶ Run generation",
            type="primary",
            disabled=job.running,
            width="stretch",
        )
    with run_cols[1]:
        summary_bits = []
        if composition.strip(): summary_bits.append(f"**{composition.strip()}**")
        if spacegroup.strip(): summary_bits.append(f"SG `{spacegroup.strip()}`")
        if crystal_systems: summary_bits.append(", ".join(crystal_systems))
        summary_bits.append(f"{int(n_generations)} samples")
        summary_bits.append(f"T={float(temperature):.1f}")
        st.markdown(
            f"<div style='color:{MUTED};padding-top:8px;font-size:0.9rem'>"
            + "  ·  ".join(summary_bits) + "</div>",
            unsafe_allow_html=True,
        )

    if run_generation:
        start_generation_job(
            job=job,
            checkpoint_path=checkpoint_path,
            device=device,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            uploaded_name=uploaded_name,
            uploaded_bytes=uploaded_bytes,
            q_min=float(q_min),
            q_max=float(q_max),
            interpolation_points=int(interpolation_points),
            wavelength_text=wavelength_text,
            n_generations=int(n_generations),
            composition=composition.strip() or None,
            spacegroup=spacegroup.strip() or None,
            crystal_systems=crystal_systems or None,
        )
        st.rerun()

    if job.running or job.finished or job.groups:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        render_job_status(job, pipeline.exp_q, pipeline.exp_i, q_min=float(q_min), q_max=float(q_max))
    else:
        st.info("Set your constraints above and click **Run generation** to sample structures.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Save ----------
    if save_clicked:
        try:
            filename = save_named_session(
                session_name=session_name,
                payload=build_session_payload(
                    pipeline=pipeline, job=job,
                    uploaded_name=uploaded_name, uploaded_bytes=uploaded_bytes,
                ),
            )
            st.toast(f"Saved session → {filename}", icon="💾")
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
