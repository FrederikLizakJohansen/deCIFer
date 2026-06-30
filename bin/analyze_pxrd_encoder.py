#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

from bin.pretrain_pxrd_encoder import (
    ContrastivePxrdModel,
    PxrdEncoderPretrainConfig,
    cap_peak_list,
    effective_qstep,
    make_condition,
    move_to_device,
    xrd_kwargs,
)
from decifer.pxrd import discrete_to_continuous_xrd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a trained contrastive PXRD encoder checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to pxrd_encoder_pretrain.pt")
    parser.add_argument("--dataset-dir", default="data/noma", help="Dataset directory containing serialized/{split}.h5")
    parser.add_argument("--split", default="val", help="Serialized split to analyze")
    parser.add_argument("--out-dir", default="", help="Output directory; defaults beside the checkpoint")
    parser.add_argument("--metrics-csv", default="", help="Optional contrastive_metrics.csv path")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum samples to embed; 0 means all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pair-samples", type=int, default=20000, help="Random pairs for PXRD/embedding correlation")
    parser.add_argument("--hard-negative-top-n", type=int, default=100)
    parser.add_argument("--tsne", action="store_true", help="Also run t-SNE; PCA is always run")
    parser.add_argument("--metadata-fields", default="crystal_system,spacegroup,cif_name", help="Comma-separated optional HDF5 fields")
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_config_from_checkpoint(checkpoint):
    model_args = checkpoint.get("model_args", {})
    return OmegaConf.merge(OmegaConf.structured(PxrdEncoderPretrainConfig()), OmegaConf.create(model_args))


def build_model(checkpoint, config, device):
    model = ContrastivePxrdModel(config).to(device)
    missing, unexpected = model.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"encoder state mismatch: missing={missing}, unexpected={unexpected}")
    if "projector_state" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector_state"], strict=True)
    model.eval()
    return model


def decode_h5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_path(dataset_dir, split):
    return os.path.join(dataset_dir, "serialized", f"{split}.h5")


def read_pxrd_rows(path, max_samples, seed, metadata_fields):
    rows = []
    with h5py.File(path, "r") as h5:
        q_key = "xrd_disc.q" if "xrd_disc.q" in h5 else "xrd.q"
        iq_key = "xrd_disc.iq" if "xrd_disc.iq" in h5 else "xrd.iq"
        n_total = len(h5[q_key])
        indices = np.arange(n_total)
        if max_samples and max_samples < n_total:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(indices, size=max_samples, replace=False))

        present_metadata = [field for field in metadata_fields if field in h5]
        for index in indices:
            row = {
                "index": int(index),
                "xrd.q": torch.tensor(np.asarray(h5[q_key][index], dtype=np.float32)),
                "xrd.iq": torch.tensor(np.asarray(h5[iq_key][index], dtype=np.float32)),
            }
            for field in present_metadata:
                row[field] = decode_h5_value(h5[field][index])
            rows.append(row)
    return rows, n_total


def clean_xrd_kwargs(config):
    fwhm = 0.5 * (float(config.fwhm_range_min) + float(config.fwhm_range_max))
    eta = 0.5 * (float(config.eta_range_min) + float(config.eta_range_max))
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


def collate_rows(rows):
    batch = {
        "index": [row["index"] for row in rows],
        "xrd.q": pad_sequence([row["xrd.q"] for row in rows], batch_first=True, padding_value=0.0),
        "xrd.iq": pad_sequence([row["xrd.iq"] for row in rows], batch_first=True, padding_value=0.0),
    }
    for key in rows[0]:
        if key not in batch and key not in {"xrd.q", "xrd.iq"}:
            batch[key] = [row.get(key) for row in rows]
    return batch


def make_clean_condition(batch_q, batch_iq, config, kwargs):
    if config.condition_encoder in {"peak", "hybrid"}:
        peak_q, peak_iq = cap_peak_list(batch_q, batch_iq, config.max_peak_list_peaks)
    if config.condition_encoder == "peak":
        return {"peak_q": peak_q, "peak_iq": peak_iq}
    dense = discrete_to_continuous_xrd(batch_q, batch_iq, **kwargs)["iq"]
    if config.condition_encoder == "hybrid":
        return {"dense": dense, "peak_q": peak_q, "peak_iq": peak_iq}
    return dense


@torch.no_grad()
def embed_rows(model, config, rows, batch_size, device, use_training_augmentations=False):
    projected = []
    pooled = []
    dense_patterns = []
    clean_kwargs = clean_xrd_kwargs(config)
    aug_kwargs = xrd_kwargs(config)
    for start in range(0, len(rows), batch_size):
        batch = collate_rows(rows[start:start + batch_size])
        batch_q = batch["xrd.q"].to(device)
        batch_iq = batch["xrd.iq"].to(device)
        if use_training_augmentations:
            condition = make_condition(batch_q, batch_iq, config, aug_kwargs)
        else:
            condition = make_clean_condition(batch_q, batch_iq, config, clean_kwargs)
        condition = move_to_device(condition, device)
        tokens = model.encoder(condition)
        pooled_batch = torch.nn.functional.normalize(tokens.mean(dim=1), dim=-1)
        projected_batch = model(condition)
        projected.append(projected_batch.cpu())
        pooled.append(pooled_batch.cpu())

        dense = discrete_to_continuous_xrd(batch_q, batch_iq, **clean_kwargs)["iq"]
        dense_patterns.append(dense.cpu())
    return {
        "projected": torch.cat(projected, dim=0).numpy(),
        "pooled": torch.cat(pooled, dim=0).numpy(),
        "dense_iq": torch.cat(dense_patterns, dim=0).numpy(),
    }


def normalized(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearmanr(x, y):
    return pearsonr(rankdata(x), rankdata(y))


def rwp(reference, candidate):
    return float(np.sqrt(np.sum((reference - candidate) ** 2) / np.clip(np.sum(reference ** 2), 1e-12, None)))


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def save_figure(fig, out_dir, stem):
    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}


def load_metric_rows(metrics_csv):
    if not metrics_csv or not os.path.exists(metrics_csv):
        return []
    with open(metrics_csv, newline="") as f:
        return [
            {
                key: float(value) if key != "iteration" else int(float(value))
                for key, value in row.items()
                if value not in {"", None}
            }
            for row in csv.DictReader(f)
        ]


def plot_training_metrics(metrics_rows, out_dir):
    if not metrics_rows:
        return None
    plt = configure_matplotlib()
    iterations = [row["iteration"] for row in metrics_rows]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.8), sharex=True)
    axes[0].plot(iterations, [row["loss"] for row in metrics_rows], color="#1f77b4", linewidth=1.3)
    axes[0].set_ylabel("NT-Xent loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(iterations, [row["positive_similarity"] for row in metrics_rows], label="positive", color="#2ca02c", linewidth=1.2)
    axes[1].plot(iterations, [row["negative_similarity"] for row in metrics_rows], label="negative", color="#d62728", linewidth=1.2)
    axes[1].plot(iterations, [row["similarity_margin"] for row in metrics_rows], label="margin", color="#9467bd", linewidth=1.0)
    axes[1].set_ylabel("cosine similarity")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    axes[2].plot(iterations, [row["retrieval_top1"] for row in metrics_rows], color="#ff7f0e", linewidth=1.3)
    axes[2].set_ylabel("retrieval top-1")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Contrastive PXRD encoder pretraining")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "training_metrics")
    plt.close(fig)
    return paths


def compute_reductions(embeddings, run_tsne, seed):
    from sklearn.decomposition import PCA

    reductions = {}
    pca_model = PCA(n_components=min(10, embeddings.shape[1]), random_state=seed)
    pca = pca_model.fit_transform(embeddings)
    reductions["pca2"] = pca[:, :2]
    explained = pca_model.explained_variance_ratio_
    if run_tsne and len(embeddings) >= 5:
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
        reductions["tsne2"] = tsne.fit_transform(embeddings)
    return reductions, explained


def label_array(rows, field):
    if not rows or field not in rows[0]:
        return None
    values = [row.get(field) for row in rows]
    if all(value is None for value in values):
        return None
    return np.asarray(values)


def plot_embedding(coords, labels, label_name, title, out_dir, stem):
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.75, linewidths=0, color="#4c78a8")
    else:
        _, numeric_labels = np.unique(labels, return_inverse=True)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=numeric_labels, s=9, alpha=0.78, linewidths=0, cmap="tab20")
        unique = np.unique(labels)
        if len(unique) <= 12:
            handles = [
                ax.scatter([], [], s=18, color=scatter.cmap(scatter.norm(i)), label=str(label))
                for i, label in enumerate(unique)
            ]
            ax.legend(handles=handles, title=label_name, frameon=False, loc="best")
        else:
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label(label_name)
    ax.set_title(title)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_explained_variance(explained, out_dir, stem="pca_explained_variance"):
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained, color="#4c78a8")
    ax.plot(x, np.cumsum(explained), color="#f58518", marker="o", linewidth=1.2)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_ylim(0.0, min(1.0, max(0.1, float(np.cumsum(explained)[-1]) * 1.08)))
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def pairwise_similarity_matrix(embeddings):
    z = normalized(embeddings)
    return z @ z.T


def knn_label_agreement(similarity, labels, ks=(1, 5, 10)):
    if labels is None:
        return {}
    labels = np.asarray(labels)
    order = np.argsort(-similarity, axis=1)
    order = order[:, 1:max(ks) + 1]
    out = {}
    for k in ks:
        if k >= len(labels):
            continue
        topk = order[:, :k]
        out[f"top{k}"] = float(np.mean(labels[topk] == labels[:, None]))
    return out


def silhouette_score_or_none(embeddings, labels, seed):
    if labels is None or len(np.unique(labels)) < 2:
        return None
    if len(np.unique(labels)) >= len(labels):
        return None
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return None
    if len(embeddings) > 1000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(len(embeddings)), size=1000, replace=False)
        embeddings = embeddings[idx]
        labels = np.asarray(labels)[idx]
    if len(np.unique(labels)) < 2:
        return None
    if len(np.unique(labels)) >= len(labels):
        return None
    try:
        return float(silhouette_score(embeddings, labels, metric="cosine"))
    except ValueError:
        return None


def trustworthiness_or_none(embeddings, coords):
    try:
        from sklearn.manifold import trustworthiness
    except Exception:
        return None
    n_neighbors = min(10, max(1, len(embeddings) // 5))
    if len(embeddings) <= n_neighbors + 1:
        return None
    return float(trustworthiness(embeddings, coords, n_neighbors=n_neighbors, metric="cosine"))


def plot_similarity_heatmap(similarity, labels, label_name, out_dir, stem="embedding_similarity_heatmap"):
    plt = configure_matplotlib()
    n = min(200, similarity.shape[0])
    if labels is not None:
        order = np.lexsort((np.arange(len(labels)), labels))[:n]
    else:
        order = np.arange(n)
    matrix = similarity[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="viridis", aspect="auto")
    ax.set_title("Encoder cosine similarity")
    ax.set_xlabel("sample")
    ax.set_ylabel("sample")
    if labels is not None:
        ax.text(0.02, 0.98, f"sorted by {label_name}", transform=ax.transAxes, ha="left", va="top", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def sample_pair_indices(n, pair_samples, seed):
    rng = np.random.default_rng(seed)
    if n < 2 or pair_samples <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    i = rng.integers(0, n, size=pair_samples)
    j = rng.integers(0, n - 1, size=pair_samples)
    j = np.where(j >= i, j + 1, j)
    return i, j


def pair_correlation_stats(embeddings, dense_iq, pair_samples, seed):
    i, j = sample_pair_indices(len(embeddings), pair_samples, seed)
    if len(i) == 0:
        return {}, None
    z = normalized(embeddings)
    pxrd = normalized(dense_iq)
    embedding_cosine = np.sum(z[i] * z[j], axis=1)
    pxrd_cosine = np.sum(pxrd[i] * pxrd[j], axis=1)
    rwps = np.asarray([rwp(dense_iq[a], dense_iq[b]) for a, b in zip(i, j)])
    stats = {
        "n_pairs": int(len(i)),
        "embedding_vs_pxrd_cosine_pearson": pearsonr(embedding_cosine, pxrd_cosine),
        "embedding_vs_pxrd_cosine_spearman": spearmanr(embedding_cosine, pxrd_cosine),
        "embedding_cosine_vs_rwp_pearson": pearsonr(embedding_cosine, rwps),
        "embedding_cosine_vs_rwp_spearman": spearmanr(embedding_cosine, rwps),
    }
    pairs = {
        "embedding_cosine": embedding_cosine,
        "pxrd_cosine": pxrd_cosine,
        "rwp": rwps,
    }
    return stats, pairs


def plot_pair_correlation(pairs, stats, out_dir, stem="pxrd_similarity_vs_embedding_similarity"):
    if pairs is None:
        return None
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].scatter(pairs["pxrd_cosine"], pairs["embedding_cosine"], s=4, alpha=0.12, linewidths=0, color="#4c78a8")
    axes[0].set_xlabel("raw PXRD cosine")
    axes[0].set_ylabel("encoder cosine")
    axes[0].set_title(f"Pearson r={stats['embedding_vs_pxrd_cosine_pearson']:.3f}")
    axes[0].grid(alpha=0.2)
    axes[1].scatter(pairs["rwp"], pairs["embedding_cosine"], s=4, alpha=0.12, linewidths=0, color="#f58518")
    axes[1].set_xlabel("pairwise Rwp")
    axes[1].set_ylabel("encoder cosine")
    axes[1].set_title(f"Spearman rho={stats['embedding_cosine_vs_rwp_spearman']:.3f}")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


@torch.no_grad()
def augmentation_invariance(model, config, rows, batch_size, device, embedding="projected"):
    first = embed_rows(model, config, rows, batch_size, device, use_training_augmentations=True)[embedding]
    second = embed_rows(model, config, rows, batch_size, device, use_training_augmentations=True)[embedding]
    first = normalized(first)
    second = normalized(second)
    same = np.sum(first * second, axis=1)
    rng = np.random.default_rng(int(config.seed))
    perm = rng.permutation(len(rows))
    if np.any(perm == np.arange(len(rows))):
        perm = np.roll(perm, 1)
    random_pair = np.sum(first * second[perm], axis=1)
    return {
        "same": same,
        "random": random_pair,
        "summary": {
            "same_mean": float(np.mean(same)),
            "same_std": float(np.std(same)),
            "random_mean": float(np.mean(random_pair)),
            "random_std": float(np.std(random_pair)),
            "margin_mean": float(np.mean(same - random_pair)),
        },
    }


def plot_augmentation_invariance(invariance, out_dir, stem="augmentation_invariance"):
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bins = np.linspace(-1, 1, 60)
    ax.hist(invariance["random"], bins=bins, density=True, alpha=0.55, color="#d62728", label="random pair")
    ax.hist(invariance["same"], bins=bins, density=True, alpha=0.55, color="#2ca02c", label="same structure, two views")
    ax.set_xlabel("encoder cosine")
    ax.set_ylabel("density")
    ax.set_title("Augmentation invariance")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def hard_negatives(similarity, rows, labels, label_name, top_n):
    if labels is None or top_n <= 0:
        return []
    labels = np.asarray(labels)
    rows_out = []
    for i in range(len(labels)):
        candidates = np.where(labels != labels[i])[0]
        if len(candidates) == 0:
            continue
        j = candidates[np.argmax(similarity[i, candidates])]
        rows_out.append({
            "sample_index": rows[i]["index"],
            "neighbor_index": rows[j]["index"],
            "sample_name": rows[i].get("cif_name", ""),
            "neighbor_name": rows[j].get("cif_name", ""),
            f"sample_{label_name}": labels[i],
            f"neighbor_{label_name}": labels[j],
            "encoder_cosine": float(similarity[i, j]),
        })
    rows_out.sort(key=lambda row: row["encoder_cosine"], reverse=True)
    return rows_out[:top_n]


def representative_samples(similarity, rows, labels, label_name):
    if labels is None:
        return []
    labels = np.asarray(labels)
    out = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        within = similarity[np.ix_(idx, idx)].mean(axis=1)
        chosen = idx[int(np.argmax(within))]
        out.append({
            label_name: label,
            "sample_index": rows[chosen]["index"],
            "sample_name": rows[chosen].get("cif_name", ""),
            "mean_within_label_cosine": float(np.max(within)),
            "n_label_samples": int(len(idx)),
        })
    return out


def plot_pxrd_panel(rows, dense_iq, config, labels, label_name, out_dir, seed):
    plt = configure_matplotlib()
    rng = np.random.default_rng(seed)
    n = min(12, len(rows))
    chosen = rng.choice(np.arange(len(rows)), size=n, replace=False)
    q_grid = torch.arange(config.qmin, config.qmax, effective_qstep(config), dtype=torch.float32).numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for offset, idx in enumerate(chosen):
        label_text = ""
        if labels is not None:
            label_text = f", {label_name}={labels[idx]}"
        ax.plot(q_grid, dense_iq[idx] + offset * 1.15, linewidth=0.9, label=f"{rows[idx].get('cif_name', rows[idx]['index'])}{label_text}")
    ax.set_xlabel("Q")
    ax.set_ylabel("normalized intensity + offset")
    ax.set_title("Sample PXRD traces used for encoder analysis")
    if n <= 8:
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "sample_pxrd_traces")
    plt.close(fig)
    return paths


def write_sample_summary(path, rows):
    metadata_keys = [key for key in rows[0] if key not in {"xrd.q", "xrd.iq"}]
    out_rows = []
    for row in rows:
        out = {key: row.get(key, "") for key in metadata_keys}
        out["n_peaks"] = int((row["xrd.q"] != 0).sum().item())
        out["q_min"] = float(row["xrd.q"].min().item()) if len(row["xrd.q"]) else float("nan")
        out["q_max"] = float(row["xrd.q"].max().item()) if len(row["xrd.q"]) else float("nan")
        out_rows.append(out)
    save_csv(path, out_rows, list(out_rows[0].keys()))


def default_out_dir(checkpoint_path, split):
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    return os.path.join(checkpoint_dir, f"encoder_analysis_{split}")


def prefixed_stem(prefix, stem):
    return stem if not prefix else f"{prefix}_{stem}"


def analyze_embedding_space(space_name, embeddings, dense_iq, rows, args, out_dir, primary_label_name):
    prefix = "" if space_name == "projected" else space_name
    reductions, explained = compute_reductions(embeddings, args.tsne, args.seed)
    similarity = pairwise_similarity_matrix(embeddings)
    primary_labels = label_array(rows, primary_label_name)

    figure_paths = {
        "pca_explained_variance": plot_explained_variance(
            explained,
            out_dir,
            prefixed_stem(prefix, "pca_explained_variance"),
        ),
        "similarity_heatmap": plot_similarity_heatmap(
            similarity,
            primary_labels,
            primary_label_name,
            out_dir,
            prefixed_stem(prefix, "embedding_similarity_heatmap"),
        ),
    }

    label_metrics: Dict[str, Dict] = {}
    for label_name in ["crystal_system", "spacegroup"]:
        labels = label_array(rows, label_name)
        if labels is None:
            continue
        label_metrics[label_name] = {
            "n_classes": int(len(np.unique(labels))),
            "knn_label_agreement": knn_label_agreement(similarity, labels),
            "silhouette_cosine": silhouette_score_or_none(embeddings, labels, args.seed),
        }
        hard = hard_negatives(similarity, rows, labels, label_name, args.hard_negative_top_n)
        if hard:
            save_csv(
                os.path.join(out_dir, prefixed_stem(prefix, f"hard_negatives_by_{label_name}.csv")),
                hard,
                list(hard[0].keys()),
            )
        reps = representative_samples(similarity, rows, labels, label_name)
        if reps:
            save_csv(
                os.path.join(out_dir, prefixed_stem(prefix, f"representative_samples_by_{label_name}.csv")),
                reps,
                list(reps[0].keys()),
            )
        for reduction_name, coords in reductions.items():
            figure_paths[f"{reduction_name}_{label_name}"] = plot_embedding(
                coords,
                labels,
                label_name,
                f"{reduction_name.upper()} {space_name} PXRD encoder space colored by {label_name}",
                out_dir,
                prefixed_stem(prefix, f"{reduction_name}_by_{label_name}"),
            )

    if not label_metrics:
        for reduction_name, coords in reductions.items():
            figure_paths[reduction_name] = plot_embedding(
                coords,
                None,
                "",
                f"{reduction_name.upper()} {space_name} PXRD encoder space",
                out_dir,
                prefixed_stem(prefix, reduction_name),
            )

    reduction_metrics = {
        name: {"trustworthiness": trustworthiness_or_none(embeddings, coords)}
        for name, coords in reductions.items()
    }
    pair_stats, pairs = pair_correlation_stats(embeddings, dense_iq, args.pair_samples, args.seed)
    figure_paths["pair_correlation"] = plot_pair_correlation(
        pairs,
        pair_stats,
        out_dir,
        prefixed_stem(prefix, "pxrd_similarity_vs_embedding_similarity"),
    )
    return {
        "embedding_dim": int(embeddings.shape[1]),
        "pca_explained_variance_ratio": [float(value) for value in explained],
        "pca_explained_variance_ratio_cumulative": [float(value) for value in np.cumsum(explained)],
        "label_metrics": label_metrics,
        "reduction_metrics": reduction_metrics,
        "pair_correlation": pair_stats,
        "figure_paths": figure_paths,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir or default_out_dir(args.checkpoint, args.split)
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = load_config_from_checkpoint(checkpoint)
    if "seed" in config:
        config.seed = args.seed
    model = build_model(checkpoint, config, device)

    h5_path = dataset_path(args.dataset_dir, args.split)
    metadata_fields = [field.strip() for field in args.metadata_fields.split(",") if field.strip()]
    rows, n_total = read_pxrd_rows(h5_path, args.max_samples, args.seed, metadata_fields)
    if not rows:
        raise ValueError(f"no rows loaded from {h5_path}")

    print(f"Loaded {len(rows)} / {n_total} samples from {h5_path}", flush=True)
    encoded = embed_rows(model, config, rows, args.batch_size, device, use_training_augmentations=False)
    embeddings = encoded["projected"]
    pooled = encoded["pooled"]
    dense_iq = encoded["dense_iq"]

    np.savez_compressed(
        os.path.join(out_dir, "encoder_embeddings.npz"),
        projected=embeddings,
        pooled=pooled,
        dense_iq=dense_iq,
        source_indices=np.asarray([row["index"] for row in rows], dtype=np.int64),
    )
    write_sample_summary(os.path.join(out_dir, "sample_summary.csv"), rows)

    metrics_csv = args.metrics_csv
    if not metrics_csv:
        metrics_csv = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "contrastive_metrics.csv")
    metric_rows = load_metric_rows(metrics_csv)

    primary_label_name = "crystal_system" if label_array(rows, "crystal_system") is not None else "spacegroup"
    primary_labels = label_array(rows, primary_label_name)

    figure_paths = {
        "training_metrics": plot_training_metrics(metric_rows, out_dir),
        "sample_pxrd_traces": plot_pxrd_panel(rows, dense_iq, config, primary_labels, primary_label_name, out_dir, args.seed),
    }
    embedding_spaces = {
        "projected": analyze_embedding_space("projected", embeddings, dense_iq, rows, args, out_dir, primary_label_name),
        "pooled": analyze_embedding_space("pooled", pooled, dense_iq, rows, args, out_dir, primary_label_name),
    }
    figure_paths.update({
        f"{space_name}_{key}": value
        for space_name, space_summary in embedding_spaces.items()
        for key, value in space_summary["figure_paths"].items()
    })
    invariance_by_space = {
        "projected": augmentation_invariance(model, config, rows, args.batch_size, device, "projected"),
        "pooled": augmentation_invariance(model, config, rows, args.batch_size, device, "pooled"),
    }
    figure_paths["augmentation_invariance"] = plot_augmentation_invariance(invariance_by_space["projected"], out_dir)
    figure_paths["pooled_augmentation_invariance"] = plot_augmentation_invariance(
        invariance_by_space["pooled"],
        out_dir,
        "pooled_augmentation_invariance",
    )

    final_training_metrics = metric_rows[-1] if metric_rows else {}
    projected_summary = embedding_spaces["projected"]
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset_path": os.path.abspath(h5_path),
        "out_dir": os.path.abspath(out_dir),
        "n_total_split_samples": int(n_total),
        "n_analyzed_samples": int(len(rows)),
        "embedding_dim": projected_summary["embedding_dim"],
        "condition_encoder": str(config.condition_encoder),
        "final_training_metrics": final_training_metrics,
        "pca_explained_variance_ratio": projected_summary["pca_explained_variance_ratio"],
        "pca_explained_variance_ratio_cumulative": projected_summary["pca_explained_variance_ratio_cumulative"],
        "label_metrics": projected_summary["label_metrics"],
        "reduction_metrics": projected_summary["reduction_metrics"],
        "pair_correlation": projected_summary["pair_correlation"],
        "embedding_spaces": embedding_spaces,
        "augmentation_invariance": invariance_by_space["projected"]["summary"],
        "augmentation_invariance_by_space": {
            space_name: space_invariance["summary"]
            for space_name, space_invariance in invariance_by_space.items()
        },
        "figure_paths": figure_paths,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json(os.path.join(out_dir, "analysis_summary.json"), summary)
    save_json(os.path.join(out_dir, "figure_manifest.json"), figure_paths)

    print(json.dumps({
        "out_dir": os.path.abspath(out_dir),
        "n_analyzed_samples": len(rows),
        "final_loss": final_training_metrics.get("loss"),
        "projected": {
            "label_metrics": embedding_spaces["projected"]["label_metrics"],
            "pair_correlation": embedding_spaces["projected"]["pair_correlation"],
        },
        "pooled": {
            "label_metrics": embedding_spaces["pooled"]["label_metrics"],
            "pair_correlation": embedding_spaces["pooled"]["pair_correlation"],
        },
        "augmentation_invariance": {
            space_name: space_invariance["summary"]
            for space_name, space_invariance in invariance_by_space.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
