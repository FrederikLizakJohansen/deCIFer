#!/usr/bin/env python3

import argparse
import csv
import faulthandler
import json
import math
import os
import random
import signal
import time
import yaml
from dataclasses import asdict, dataclass, field
from typing import List, Optional

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

faulthandler.enable()

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler

from decifer.decifer_model import DeciferConfig, build_condition_encoder
from decifer.pxrd import discrete_to_continuous_xrd, nyquist_qstep


METRIC_FIELDS = [
    "iteration",
    "loss",
    "contrastive_loss",
    "pxrd_similarity_loss",
    "crystal_system_loss",
    "spacegroup_loss",
    "positive_similarity",
    "negative_similarity",
    "similarity_margin",
    "retrieval_top1",
    "crystal_system_accuracy",
    "spacegroup_accuracy",
    "time_seconds",
]

STOP_REQUESTED = False
STOP_SIGNAL = None


@dataclass
class PxrdEncoderPretrainConfig:
    out_dir: str = "pxrd_encoder_pretrain"
    dataset: str = "data/noma"
    split: str = "train"
    batch_size: int = 128
    num_workers_dataloader: int = 4
    dataloader_multiprocessing_context: str = "spawn"
    dataloader_timeout_seconds: int = 0
    pin_memory: bool = False
    preload_dataset_to_memory: bool = False
    synthetic_debug_data: bool = False
    synthetic_debug_size: int = 256
    max_raw_peaks_per_sample: int = 2048
    resume: bool = False
    resume_from: str = ""
    save_on_interrupt: bool = True
    max_runtime_seconds: int = 0
    max_iters: int = 10_000
    eval_interval: int = 500
    log_interval: int = 20
    plot_interval: int = 20
    plot_window: int = 500
    live_plot: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    temperature: float = 0.1
    projection_dim: int = 128
    contrastive_embedding: str = "projected"
    pxrd_similarity_loss_weight: float = 0.0
    pxrd_similarity_temperature: float = 0.1
    pxrd_similarity_target_temperature: float = 0.1
    crystal_system_loss_weight: float = 0.0
    spacegroup_loss_weight: float = 0.0
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"

    condition_encoder: str = "conv"
    condition_size: int = 1000
    condition_n_tokens: int = 16
    dense_condition_n_tokens: int = 16
    peak_condition_n_tokens: int = 16
    peak_encoder_hidden_dim: int = 128
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.0
    bias: bool = False
    pxrd_encoder_channels: int = 64
    pxrd_encoder_kernel_size: int = 7
    condition_embedder_hidden_layers: List[int] = field(default_factory=lambda: [256])

    qmin: float = 0.0
    qmax: float = 10.0
    qstep: float = 0.01
    nyquist_points_per_fwhm: float = 0.0
    fwhm_range_min: float = 0.03
    fwhm_range_max: float = 0.09
    eta_range_min: float = 0.5
    eta_range_max: float = 0.5
    noise_range_min: float = 0.0
    noise_range_max: float = 0.03
    intensity_scale_range_min: float = 0.9
    intensity_scale_range_max: float = 1.0
    mask_prob: float = 0.0
    q_shift_range_min: float = -0.02
    q_shift_range_max: float = 0.02
    q_scale_range_min: float = 0.998
    q_scale_range_max: float = 1.002
    peak_intensity_jitter_range_min: float = 0.9
    peak_intensity_jitter_range_max: float = 1.1
    peak_dropout_prob: float = 0.0
    background_range_min: float = 0.0
    background_range_max: float = 0.03
    impurity_peak_count_min: int = 0
    impurity_peak_count_max: int = 0
    impurity_intensity_range_min: float = 0.01
    impurity_intensity_range_max: float = 0.05
    particle_size_range_min: float = 0.0
    particle_size_range_max: float = 0.0
    peak_asymmetry_range_min: float = 0.0
    peak_asymmetry_range_max: float = 0.0
    final_normalize_xrd: bool = True
    max_xrd_peaks: int = 1024
    max_peak_list_peaks: int = 512


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)
    config = OmegaConf.merge(OmegaConf.structured(PxrdEncoderPretrainConfig()), OmegaConf.create(yaml_config))
    os.makedirs(config.out_dir, exist_ok=True)
    return config


def request_stop(signum, _frame):
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = signum
    print(f"Stop requested by signal {signum}; saving at the next safe checkpoint point.", flush=True)


def install_signal_handlers():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def checkpoint_path(config):
    return os.path.join(config.out_dir, "pxrd_encoder_pretrain.pt")


def requested_resume_path(config):
    if config.resume_from:
        return config.resume_from
    if config.resume:
        return checkpoint_path(config)
    return ""


def resolve_dtype(config, device):
    if config.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("CUDA device does not report bfloat16 support; falling back to float16.", flush=True)
        return torch.float16
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]


def dataloader_kwargs(config, device):
    kwargs = {
        "num_workers": config.num_workers_dataloader,
        "collate_fn": make_collate_fn(config),
        "pin_memory": bool(config.pin_memory and device.type == "cuda"),
        "drop_last": True,
        "timeout": config.dataloader_timeout_seconds if config.num_workers_dataloader > 0 else 0,
    }
    if config.num_workers_dataloader > 0:
        context = str(config.dataloader_multiprocessing_context or "").strip()
        if context:
            kwargs["multiprocessing_context"] = context
    return kwargs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_cuda(seed, device):
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def effective_qstep(config):
    if config.nyquist_points_per_fwhm > 0:
        return nyquist_qstep(config.fwhm_range_min, config.nyquist_points_per_fwhm)
    return config.qstep


def range_or_none(range_min, range_max, identity):
    if range_min == identity and range_max == identity:
        return None
    return (range_min, range_max)


def int_range_or_none(range_min, range_max, identity):
    if range_min == identity and range_max == identity:
        return None
    return (range_min, range_max)


def xrd_kwargs(config):
    return {
        "qmin": config.qmin,
        "qmax": config.qmax,
        "qstep": effective_qstep(config),
        "nyquist_points_per_fwhm": None,
        "fwhm_range": (config.fwhm_range_min, config.fwhm_range_max),
        "eta_range": (config.eta_range_min, config.eta_range_max),
        "noise_range": range_or_none(config.noise_range_min, config.noise_range_max, 0.0),
        "intensity_scale_range": range_or_none(config.intensity_scale_range_min, config.intensity_scale_range_max, 1.0),
        "mask_prob": config.mask_prob,
        "q_shift_range": range_or_none(config.q_shift_range_min, config.q_shift_range_max, 0.0),
        "q_scale_range": range_or_none(config.q_scale_range_min, config.q_scale_range_max, 1.0),
        "peak_intensity_jitter_range": range_or_none(config.peak_intensity_jitter_range_min, config.peak_intensity_jitter_range_max, 1.0),
        "peak_dropout_prob": config.peak_dropout_prob,
        "background_range": range_or_none(config.background_range_min, config.background_range_max, 0.0),
        "impurity_peak_count_range": int_range_or_none(config.impurity_peak_count_min, config.impurity_peak_count_max, 0),
        "impurity_intensity_range": (config.impurity_intensity_range_min, config.impurity_intensity_range_max),
        "particle_size_range": range_or_none(config.particle_size_range_min, config.particle_size_range_max, 0.0),
        "peak_asymmetry_range": range_or_none(config.peak_asymmetry_range_min, config.peak_asymmetry_range_max, 0.0),
        "final_normalize": config.final_normalize_xrd,
        "max_peaks": config.max_xrd_peaks if config.max_xrd_peaks > 0 else None,
    }


def model_config(config):
    return DeciferConfig(
        condition=True,
        condition_size=len(np.arange(config.qmin, config.qmax, effective_qstep(config))),
        condition_encoder=config.condition_encoder,
        condition_n_tokens=config.condition_n_tokens,
        dense_condition_n_tokens=config.dense_condition_n_tokens,
        peak_condition_n_tokens=config.peak_condition_n_tokens,
        peak_encoder_hidden_dim=config.peak_encoder_hidden_dim,
        condition_qmin=config.qmin,
        condition_qmax=config.qmax,
        n_embd=config.n_embd,
        n_head=config.n_head,
        dropout=config.dropout,
        bias=config.bias,
        pxrd_encoder_channels=config.pxrd_encoder_channels,
        pxrd_encoder_kernel_size=config.pxrd_encoder_kernel_size,
        condition_embedder_hidden_layers=config.condition_embedder_hidden_layers,
    )


class ContrastivePxrdModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = build_condition_encoder(model_config(config))
        self.projector = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_embd, config.projection_dim, bias=config.bias),
        )
        self.crystal_system_head = nn.Linear(config.n_embd, 7, bias=config.bias)
        self.spacegroup_head = nn.Linear(config.n_embd, 230, bias=config.bias)

    def encode(self, condition):
        tokens = self.encoder(condition)
        pooled = nn.functional.normalize(tokens.mean(dim=1), dim=-1)
        projected = nn.functional.normalize(self.projector(pooled), dim=-1)
        return {"tokens": tokens, "pooled": pooled, "projected": projected}

    def forward(self, condition, embedding: Optional[str] = None):
        encoded = self.encode(condition)
        embedding = embedding or self.config.contrastive_embedding
        if embedding not in {"projected", "pooled"}:
            raise ValueError(f"unknown contrastive_embedding: {embedding}")
        return encoded[embedding]


class InMemoryPxrdDataset(Dataset):
    def __init__(self, h5_path, data_keys):
        from decifer.decifer_dataset import DeciferDataset

        source = DeciferDataset(h5_path, data_keys, lazy_open=False)
        try:
            self.rows = []
            for index in range(len(source)):
                row = source[index]
                self.rows.append({key: clone_dataset_value(value) for key, value in row.items()})
        finally:
            source.close()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class SyntheticPxrdDataset(Dataset):
    def __init__(self, size, qmin, qmax, seed):
        self.rows = []
        rng = np.random.default_rng(seed)
        for _ in range(size):
            n_peaks = int(rng.integers(32, 129))
            q = np.sort(rng.uniform(qmin + 0.1, qmax - 0.1, size=n_peaks)).astype(np.float32)
            iq = rng.lognormal(mean=0.0, sigma=0.8, size=n_peaks).astype(np.float32)
            iq = iq / max(float(iq.max()), 1e-6)
            self.rows.append({
                "xrd.q": torch.tensor(q, dtype=torch.float32),
                "xrd.iq": torch.tensor(iq, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def clone_dataset_value(value):
    if torch.is_tensor(value):
        return value.clone()
    return value


def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(2 * batch_size, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat((labels + batch_size, labels), dim=0)
    return nn.functional.cross_entropy(logits, labels)


def soft_cross_entropy(logits, targets):
    return -(targets * nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def pxrd_similarity_loss(z1, z2, dense_iq, logit_temperature, target_temperature):
    dense = nn.functional.normalize(dense_iq, dim=-1)
    target_logits = dense @ dense.T / target_temperature
    targets = nn.functional.softmax(target_logits, dim=-1).detach()
    logits_12 = z1 @ z2.T / logit_temperature
    logits_21 = z2 @ z1.T / logit_temperature
    return 0.5 * (soft_cross_entropy(logits_12, targets) + soft_cross_entropy(logits_21, targets.T))


def label_loss(logits_1, logits_2, labels):
    return 0.5 * (
        nn.functional.cross_entropy(logits_1, labels)
        + nn.functional.cross_entropy(logits_2, labels)
    )


def clean_xrd_kwargs(config):
    fwhm = 0.5 * (config.fwhm_range_min + config.fwhm_range_max)
    eta = 0.5 * (config.eta_range_min + config.eta_range_max)
    return {
        "qmin": config.qmin,
        "qmax": config.qmax,
        "qstep": effective_qstep(config),
        "nyquist_points_per_fwhm": None,
        "fwhm_range": (fwhm, fwhm),
        "eta_range": (eta, eta),
        "noise_range": None,
        "intensity_scale_range": None,
        "mask_prob": None,
        "q_shift_range": None,
        "q_scale_range": None,
        "peak_intensity_jitter_range": None,
        "peak_dropout_prob": None,
        "background_range": None,
        "impurity_peak_count_range": None,
        "particle_size_range": None,
        "peak_asymmetry_range": None,
        "final_normalize": config.final_normalize_xrd,
        "max_peaks": config.max_xrd_peaks if config.max_xrd_peaks > 0 else None,
    }


def needs_label_fields(config):
    return config.crystal_system_loss_weight > 0 or config.spacegroup_loss_weight > 0


def dataset_fields(config):
    fields = ["xrd.q", "xrd.iq"]
    if config.crystal_system_loss_weight > 0:
        fields.append("crystal_system")
    if config.spacegroup_loss_weight > 0:
        fields.append("spacegroup")
    return fields


@torch.no_grad()
def contrastive_batch_metrics(z1, z2):
    similarities = z1 @ z2.T
    labels = torch.arange(z1.size(0), device=z1.device)
    positive_similarity = similarities.diag().mean()
    negative_mask = ~torch.eye(z1.size(0), dtype=torch.bool, device=z1.device)
    negative_similarity = similarities[negative_mask].mean()
    retrieval_1_to_2 = (similarities.argmax(dim=1) == labels).float().mean()
    retrieval_2_to_1 = (similarities.argmax(dim=0) == labels).float().mean()
    return {
        "positive_similarity": float(positive_similarity.item()),
        "negative_similarity": float(negative_similarity.item()),
        "retrieval_top1": float(0.5 * (retrieval_1_to_2.item() + retrieval_2_to_1.item())),
    }


def cap_single_peak_list(q, iq, max_peaks):
    if max_peaks <= 0 or q.numel() <= max_peaks:
        return q, iq
    valid = q != 0
    scores = iq.masked_fill(~valid, float("-inf"))
    peak_indices = torch.topk(scores, k=max_peaks).indices
    return q[peak_indices], iq[peak_indices]


def make_collate_fn(config):
    def _collate_fn(batch):
        return collate_fn(batch, config.max_raw_peaks_per_sample)
    return _collate_fn


def collate_fn(batch, max_raw_peaks_per_sample=0):
    xrd_q = []
    xrd_iq = []
    for item in batch:
        q, iq = cap_single_peak_list(item["xrd.q"], item["xrd.iq"], max_raw_peaks_per_sample)
        xrd_q.append(q)
        xrd_iq.append(iq)
    collated = {
        "xrd.q": pad_sequence(xrd_q, batch_first=True, padding_value=0.0),
        "xrd.iq": pad_sequence(xrd_iq, batch_first=True, padding_value=0.0),
    }
    for key in ("crystal_system", "spacegroup"):
        if key in batch[0]:
            collated[key] = torch.stack([item[key].long().view(()) for item in batch])
    return collated


def cap_peak_list(batch_q, batch_iq, max_peaks):
    if max_peaks <= 0 or batch_q.size(1) <= max_peaks:
        return batch_q, batch_iq
    valid = batch_q != 0
    scores = batch_iq.masked_fill(~valid, float("-inf"))
    peak_indices = torch.topk(scores, k=max_peaks, dim=1).indices
    batch_q = torch.gather(batch_q, 1, peak_indices)
    batch_iq = torch.gather(batch_iq, 1, peak_indices)
    valid = torch.gather(valid, 1, peak_indices)
    return torch.where(valid, batch_q, torch.zeros_like(batch_q)), torch.where(valid, batch_iq, torch.zeros_like(batch_iq))


def uniform_like(shape, range_, like):
    return torch.empty(*shape, dtype=like.dtype, device=like.device).uniform_(*range_)


def augment_peak_list(batch_q, batch_iq, config):
    batch_q = batch_q.clone()
    batch_iq = batch_iq.clone()
    valid = batch_q != 0
    q_scale_range = range_or_none(config.q_scale_range_min, config.q_scale_range_max, 1.0)
    if q_scale_range is not None:
        batch_q = batch_q * uniform_like((batch_q.size(0), 1), q_scale_range, batch_q)
    q_shift_range = range_or_none(config.q_shift_range_min, config.q_shift_range_max, 0.0)
    if q_shift_range is not None:
        batch_q = batch_q + uniform_like((batch_q.size(0), 1), q_shift_range, batch_q)
    intensity_scale_range = range_or_none(config.intensity_scale_range_min, config.intensity_scale_range_max, 1.0)
    if intensity_scale_range is not None:
        batch_iq = batch_iq * uniform_like((batch_iq.size(0), 1), intensity_scale_range, batch_iq)
    jitter_range = range_or_none(config.peak_intensity_jitter_range_min, config.peak_intensity_jitter_range_max, 1.0)
    if jitter_range is not None:
        batch_iq = batch_iq * uniform_like(batch_iq.shape, jitter_range, batch_iq)
    if config.peak_dropout_prob > 0:
        keep = torch.rand(batch_iq.shape, dtype=batch_iq.dtype, device=batch_iq.device) > config.peak_dropout_prob
        batch_iq = batch_iq * keep
        valid = valid & keep
    batch_q = torch.where(valid, batch_q, torch.zeros_like(batch_q))
    batch_iq = torch.where(valid, batch_iq, torch.zeros_like(batch_iq))
    return cap_peak_list(batch_q, batch_iq, config.max_peak_list_peaks)


def make_condition(batch_q, batch_iq, config, kwargs):
    if config.condition_encoder in {"peak", "hybrid"}:
        peak_q, peak_iq = augment_peak_list(batch_q, batch_iq, config)
    if config.condition_encoder == "peak":
        return {"peak_q": peak_q, "peak_iq": peak_iq}
    dense = discrete_to_continuous_xrd(batch_q, batch_iq, **kwargs)["iq"]
    if config.condition_encoder == "hybrid":
        return {"dense": dense, "peak_q": peak_q, "peak_iq": peak_iq}
    return dense


def move_to_device(value, device):
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value.to(device)


def write_metrics_header(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def ensure_metrics_file(path, resume_iteration):
    if resume_iteration <= 0 or not os.path.exists(path):
        write_metrics_header(path)
        return
    with open(path, newline="") as f:
        header = f.readline().strip().split(",")
    if header == METRIC_FIELDS:
        return
    backup_path = path + f".legacy_{int(time.time())}"
    os.replace(path, backup_path)
    print(f"Existing metric header is incompatible; moved old metrics to {backup_path}", flush=True)
    write_metrics_header(path)


def append_metric(path, latest_path, row):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerow(row)
    with open(latest_path, "w") as f:
        json.dump(row, f, indent=2)


def read_metric_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                key: float(value) if key != "iteration" else int(float(value))
                for key, value in row.items()
                if value not in {"", None}
            })
    return rows


def update_live_plot(metrics_path, plot_path, plot_window):
    rows = read_metric_rows(metrics_path)
    if not rows:
        return
    rows = rows[-plot_window:]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not update live plot: {exc}", flush=True)
        return

    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), dpi=140, sharex=True)
    axes[0].plot(iterations, [row["loss"] for row in rows], color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("NT-Xent loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iterations, [row["positive_similarity"] for row in rows], label="positive", color="#2ca02c", linewidth=1.5)
    axes[1].plot(iterations, [row["negative_similarity"] for row in rows], label="negative", color="#d62728", linewidth=1.5)
    axes[1].plot(iterations, [row["similarity_margin"] for row in rows], label="margin", color="#9467bd", linewidth=1.2, alpha=0.8)
    axes[1].set_ylabel("cosine sim")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(iterations, [row["retrieval_top1"] for row in rows], color="#ff7f0e", linewidth=1.5)
    axes[2].set_ylabel("retrieval top-1")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.3)

    fig.suptitle("PXRD contrastive encoder pretraining")
    fig.tight_layout()
    tmp_path = plot_path + ".tmp.png"
    try:
        fig.savefig(tmp_path)
        os.replace(tmp_path, plot_path)
    except OSError as exc:
        print(f"Could not write live plot: {exc}", flush=True)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    finally:
        plt.close(fig)


def save_checkpoint(config, model, optimizer, iteration, path=None):
    decifer_config = OmegaConf.to_container(OmegaConf.create(asdict(model_config(config))), resolve=True)
    checkpoint = {
        "encoder_state": model.encoder.state_dict(),
        "projector_state": model.projector.state_dict(),
        "crystal_system_head_state": model.crystal_system_head.state_dict(),
        "spacegroup_head_state": model.spacegroup_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "model_args": OmegaConf.to_container(config, resolve=True),
        "decifer_config": decifer_config,
    }
    path = path or checkpoint_path(config)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_resume_checkpoint(config, model, optimizer, device):
    path = requested_resume_path(config)
    if not path:
        return 0
    if not os.path.exists(path):
        if config.resume_from:
            raise FileNotFoundError(f"resume_from checkpoint not found: {path}")
        print(f"resume=True but no checkpoint exists at {path}; starting from scratch.", flush=True)
        return 0
    print(f"Resuming PXRD encoder pretraining from {path}", flush=True)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    if "projector_state" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector_state"], strict=True)
    if "crystal_system_head_state" in checkpoint:
        model.crystal_system_head.load_state_dict(checkpoint["crystal_system_head_state"], strict=False)
    if "spacegroup_head_state" in checkpoint:
        model.spacegroup_head.load_state_dict(checkpoint["spacegroup_head_state"], strict=False)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = int(checkpoint.get("iteration", 0))
    print(f"Resumed at iteration {iteration}", flush=True)
    return iteration


def main():
    config = parse_config()
    install_signal_handlers()
    if config.contrastive_embedding not in {"projected", "pooled"}:
        raise ValueError("contrastive_embedding must be 'projected' or 'pooled'")
    if config.synthetic_debug_data and needs_label_fields(config):
        raise ValueError("synthetic_debug_data does not provide crystal_system or spacegroup labels")
    seed_everything(config.seed)
    device = torch.device(config.device)
    print(f"Using device: {device}", flush=True)
    print(
        f"DataLoader workers={config.num_workers_dataloader}, "
        f"context={config.dataloader_multiprocessing_context!r}, pin_memory={config.pin_memory}, "
        f"timeout={config.dataloader_timeout_seconds}s",
        flush=True,
    )
    print(
        f"max_raw_peaks_per_sample={config.max_raw_peaks_per_sample}, "
        f"resume={config.resume}, resume_from={config.resume_from!r}",
        flush=True,
    )
    if config.num_workers_dataloader > 0 and config.dataloader_multiprocessing_context not in {"spawn", "forkserver"}:
        print("WARNING: HDF5 DataLoader workers are safest with multiprocessing context 'spawn' or 'forkserver'.", flush=True)
    dataset_path = os.path.join(config.dataset, "serialized", f"{config.split}.h5")
    if config.synthetic_debug_data:
        print(f"Using synthetic debug PXRD data: {config.synthetic_debug_size} samples", flush=True)
        dataset = SyntheticPxrdDataset(config.synthetic_debug_size, config.qmin, config.qmax, config.seed)
    elif config.preload_dataset_to_memory:
        print(f"Preloading PXRD HDF5 data into memory: {dataset_path}", flush=True)
        dataset = InMemoryPxrdDataset(dataset_path, dataset_fields(config))
    else:
        from decifer.decifer_dataset import DeciferDataset

        dataset = DeciferDataset(dataset_path, dataset_fields(config), lazy_open=config.num_workers_dataloader > 0)
    if len(dataset) < config.batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples, but batch_size is {config.batch_size} and drop_last is enabled")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(dataset),
        **dataloader_kwargs(config, device),
    )
    iterator = iter(loader)
    seed_cuda(config.seed, device)
    model = ContrastivePxrdModel(config).to(device)
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    resume_iteration = load_resume_checkpoint(config, model, optimizer, device)
    kwargs = xrd_kwargs(config)
    clean_kwargs = clean_xrd_kwargs(config)
    metrics_path = os.path.join(config.out_dir, "contrastive_metrics.csv")
    latest_metrics_path = os.path.join(config.out_dir, "latest_metrics.json")
    live_plot_path = os.path.join(config.out_dir, "contrastive_live.png")
    ensure_metrics_file(metrics_path, resume_iteration)
    with open(os.path.join(config.out_dir, "pretrain_config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(config, resolve=True), f, sort_keys=False)

    dtype = resolve_dtype(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)

    model.train()
    start_time = time.monotonic()
    if resume_iteration >= config.max_iters:
        print(f"Checkpoint iteration {resume_iteration} is already >= max_iters {config.max_iters}; nothing to do.", flush=True)
        return
    for iteration in range(resume_iteration + 1, config.max_iters + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch_q = batch["xrd.q"].to(device)
        batch_iq = batch["xrd.iq"].to(device)
        crystal_system_labels = batch.get("crystal_system")
        spacegroup_labels = batch.get("spacegroup")
        if crystal_system_labels is not None:
            crystal_system_labels = crystal_system_labels.to(device) - 1
        if spacegroup_labels is not None:
            spacegroup_labels = spacegroup_labels.to(device) - 1
        condition_1 = make_condition(batch_q, batch_iq, config, kwargs)
        condition_2 = make_condition(batch_q, batch_iq, config, kwargs)
        condition_1 = move_to_device(condition_1, device)
        condition_2 = move_to_device(condition_2, device)

        optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else torch.enable_grad()
        with context:
            encoded_1 = model.encode(condition_1)
            encoded_2 = model.encode(condition_2)
            z1 = encoded_1[config.contrastive_embedding]
            z2 = encoded_2[config.contrastive_embedding]
            contrastive_loss = nt_xent_loss(z1, z2, config.temperature)
            loss = contrastive_loss
            pxrd_loss = torch.zeros((), dtype=loss.dtype, device=device)
            crystal_system_loss = torch.zeros((), dtype=loss.dtype, device=device)
            spacegroup_loss = torch.zeros((), dtype=loss.dtype, device=device)
            if config.pxrd_similarity_loss_weight > 0:
                dense_target = discrete_to_continuous_xrd(batch_q, batch_iq, **clean_kwargs)["iq"]
                pxrd_loss = pxrd_similarity_loss(
                    z1,
                    z2,
                    dense_target,
                    config.pxrd_similarity_temperature,
                    config.pxrd_similarity_target_temperature,
                )
                loss = loss + config.pxrd_similarity_loss_weight * pxrd_loss
            if config.crystal_system_loss_weight > 0:
                if crystal_system_labels is None:
                    raise ValueError("crystal_system_loss_weight requires crystal_system in the dataset")
                crystal_logits_1 = model.crystal_system_head(encoded_1["pooled"])
                crystal_logits_2 = model.crystal_system_head(encoded_2["pooled"])
                crystal_system_loss = label_loss(crystal_logits_1, crystal_logits_2, crystal_system_labels)
                loss = loss + config.crystal_system_loss_weight * crystal_system_loss
            if config.spacegroup_loss_weight > 0:
                if spacegroup_labels is None:
                    raise ValueError("spacegroup_loss_weight requires spacegroup in the dataset")
                spacegroup_logits_1 = model.spacegroup_head(encoded_1["pooled"])
                spacegroup_logits_2 = model.spacegroup_head(encoded_2["pooled"])
                spacegroup_loss = label_loss(spacegroup_logits_1, spacegroup_logits_2, spacegroup_labels)
                loss = loss + config.spacegroup_loss_weight * spacegroup_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if iteration % config.log_interval == 0 or iteration == 1:
            metrics = contrastive_batch_metrics(z1.detach(), z2.detach())
            crystal_system_accuracy = None
            spacegroup_accuracy = None
            if config.crystal_system_loss_weight > 0:
                crystal_system_accuracy = float((crystal_logits_1.detach().argmax(dim=-1) == crystal_system_labels).float().mean().item())
            if config.spacegroup_loss_weight > 0:
                spacegroup_accuracy = float((spacegroup_logits_1.detach().argmax(dim=-1) == spacegroup_labels).float().mean().item())
            row = {
                "iteration": int(iteration),
                "loss": float(loss.item()),
                "contrastive_loss": float(contrastive_loss.item()),
                "pxrd_similarity_loss": float(pxrd_loss.item()),
                "crystal_system_loss": float(crystal_system_loss.item()),
                "spacegroup_loss": float(spacegroup_loss.item()),
                "positive_similarity": metrics["positive_similarity"],
                "negative_similarity": metrics["negative_similarity"],
                "similarity_margin": metrics["positive_similarity"] - metrics["negative_similarity"],
                "retrieval_top1": metrics["retrieval_top1"],
                "crystal_system_accuracy": crystal_system_accuracy,
                "spacegroup_accuracy": spacegroup_accuracy,
                "time_seconds": time.time(),
            }
            print(
                f"iter {iteration}: loss {row['loss']:.4f}, "
                f"contrastive {row['contrastive_loss']:.4f}, "
                f"pos {row['positive_similarity']:.3f}, neg {row['negative_similarity']:.3f}, "
                f"top1 {row['retrieval_top1']:.3f}",
                flush=True,
            )
            append_metric(metrics_path, latest_metrics_path, row)
        if config.live_plot and (iteration % config.plot_interval == 0 or iteration == 1):
            update_live_plot(metrics_path, live_plot_path, config.plot_window)
        if iteration % config.eval_interval == 0:
            save_checkpoint(config, model, optimizer, iteration)
        if STOP_REQUESTED:
            if config.save_on_interrupt:
                print(f"Saving checkpoint after stop request at iteration {iteration}.", flush=True)
                save_checkpoint(config, model, optimizer, iteration)
            break
        if config.max_runtime_seconds > 0 and time.monotonic() - start_time >= config.max_runtime_seconds:
            print(f"Reached max_runtime_seconds={config.max_runtime_seconds}; saving checkpoint at iteration {iteration}.", flush=True)
            save_checkpoint(config, model, optimizer, iteration)
            break

    else:
        save_checkpoint(config, model, optimizer, config.max_iters)


if __name__ == "__main__":
    main()
