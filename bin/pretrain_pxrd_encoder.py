#!/usr/bin/env python3

import argparse
import csv
import faulthandler
import json
import math
import os
import random
import time
import yaml
from dataclasses import asdict, dataclass, field
from typing import List

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


@dataclass
class PxrdEncoderPretrainConfig:
    out_dir: str = "pxrd_encoder_pretrain"
    dataset: str = "data/noma"
    split: str = "train"
    batch_size: int = 128
    num_workers_dataloader: int = 4
    dataloader_multiprocessing_context: str = "spawn"
    pin_memory: bool = False
    preload_dataset_to_memory: bool = False
    synthetic_debug_data: bool = False
    synthetic_debug_size: int = 256
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


def resolve_dtype(config, device):
    if config.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("CUDA device does not report bfloat16 support; falling back to float16.", flush=True)
        return torch.float16
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]


def dataloader_kwargs(config, device):
    kwargs = {
        "num_workers": config.num_workers_dataloader,
        "collate_fn": collate_fn,
        "pin_memory": bool(config.pin_memory and device.type == "cuda"),
        "drop_last": True,
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

    def forward(self, condition):
        tokens = self.encoder(condition)
        pooled = tokens.mean(dim=1)
        return nn.functional.normalize(self.projector(pooled), dim=-1)


class InMemoryPxrdDataset(Dataset):
    def __init__(self, h5_path):
        from decifer.decifer_dataset import DeciferDataset

        source = DeciferDataset(h5_path, ["xrd.q", "xrd.iq"], lazy_open=False)
        try:
            self.rows = []
            for index in range(len(source)):
                row = source[index]
                self.rows.append({
                    "xrd.q": row["xrd.q"].clone(),
                    "xrd.iq": row["xrd.iq"].clone(),
                })
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


def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(2 * batch_size, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat((labels + batch_size, labels), dim=0)
    return nn.functional.cross_entropy(logits, labels)


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


def collate_fn(batch):
    return {
        "xrd.q": pad_sequence([item["xrd.q"] for item in batch], batch_first=True, padding_value=0.0),
        "xrd.iq": pad_sequence([item["xrd.iq"] for item in batch], batch_first=True, padding_value=0.0),
    }


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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "loss",
                "positive_similarity",
                "negative_similarity",
                "similarity_margin",
                "retrieval_top1",
                "time_seconds",
            ],
        )
        writer.writeheader()


def append_metric(path, latest_path, row):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
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
    fig.savefig(tmp_path)
    plt.close(fig)
    os.replace(tmp_path, plot_path)


def save_checkpoint(config, model, optimizer, iteration):
    decifer_config = OmegaConf.to_container(OmegaConf.create(asdict(model_config(config))), resolve=True)
    checkpoint = {
        "encoder_state": model.encoder.state_dict(),
        "projector_state": model.projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "model_args": OmegaConf.to_container(config, resolve=True),
        "decifer_config": decifer_config,
    }
    torch.save(checkpoint, os.path.join(config.out_dir, "pxrd_encoder_pretrain.pt"))


def main():
    config = parse_config()
    seed_everything(config.seed)
    device = torch.device(config.device)
    print(f"Using device: {device}", flush=True)
    print(
        f"DataLoader workers={config.num_workers_dataloader}, "
        f"context={config.dataloader_multiprocessing_context!r}, pin_memory={config.pin_memory}",
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
        dataset = InMemoryPxrdDataset(dataset_path)
    else:
        from decifer.decifer_dataset import DeciferDataset

        dataset = DeciferDataset(dataset_path, ["xrd.q", "xrd.iq"], lazy_open=config.num_workers_dataloader > 0)
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
    kwargs = xrd_kwargs(config)
    metrics_path = os.path.join(config.out_dir, "contrastive_metrics.csv")
    latest_metrics_path = os.path.join(config.out_dir, "latest_metrics.json")
    live_plot_path = os.path.join(config.out_dir, "contrastive_live.png")
    write_metrics_header(metrics_path)
    with open(os.path.join(config.out_dir, "pretrain_config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(config, resolve=True), f, sort_keys=False)

    dtype = resolve_dtype(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)

    model.train()
    for iteration in range(1, config.max_iters + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch_q = batch["xrd.q"].to(device)
        batch_iq = batch["xrd.iq"].to(device)
        condition_1 = make_condition(batch_q, batch_iq, config, kwargs)
        condition_2 = make_condition(batch_q, batch_iq, config, kwargs)
        condition_1 = move_to_device(condition_1, device)
        condition_2 = move_to_device(condition_2, device)

        optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else torch.enable_grad()
        with context:
            z1 = model(condition_1)
            z2 = model(condition_2)
            loss = nt_xent_loss(z1, z2, config.temperature)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if iteration % config.log_interval == 0 or iteration == 1:
            metrics = contrastive_batch_metrics(z1.detach(), z2.detach())
            row = {
                "iteration": int(iteration),
                "loss": float(loss.item()),
                "positive_similarity": metrics["positive_similarity"],
                "negative_similarity": metrics["negative_similarity"],
                "similarity_margin": metrics["positive_similarity"] - metrics["negative_similarity"],
                "retrieval_top1": metrics["retrieval_top1"],
                "time_seconds": time.time(),
            }
            print(
                f"iter {iteration}: loss {row['loss']:.4f}, "
                f"pos {row['positive_similarity']:.3f}, neg {row['negative_similarity']:.3f}, "
                f"top1 {row['retrieval_top1']:.3f}",
                flush=True,
            )
            append_metric(metrics_path, latest_metrics_path, row)
        if config.live_plot and (iteration % config.plot_interval == 0 or iteration == 1):
            update_live_plot(metrics_path, live_plot_path, config.plot_window)
        if iteration % config.eval_interval == 0:
            save_checkpoint(config, model, optimizer, iteration)

    save_checkpoint(config, model, optimizer, config.max_iters)


if __name__ == "__main__":
    main()
