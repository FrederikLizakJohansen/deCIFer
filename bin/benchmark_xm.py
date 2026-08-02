#!/usr/bin/env python3

"""Microbenchmark Forward XM training cost on the deCIFer transformer."""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.explorative_modeling import xm_best_of_k_forward
from decifer.minicif_v2 import MinicifV2Tokenizer


def benchmark(args):
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    tokenizer = MinicifV2Tokenizer()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    baseline = build_model(args, tokenizer.vocab_size, 1).to(device)
    baseline_state = baseline.state_dict()
    idx, targets = synthetic_batch(
        args.batch_size,
        args.sequence_length,
        tokenizer.vocab_size,
        tokenizer.token_to_id["<mcif2>"],
        device,
    )
    starts = [[0] for _ in range(args.batch_size)]
    results = []
    for best_of_k in args.k:
        torch.manual_seed(args.seed)
        model = build_model(args, tokenizer.vocab_size, best_of_k).to(device)
        if best_of_k > 1:
            missing, unexpected = model.load_state_dict(
                baseline_state,
                strict=False,
            )
            if missing != ["xm_mode_embeddings.weight"] or unexpected:
                raise RuntimeError(
                    f"unexpected benchmark state mismatch: {missing}, {unexpected}"
                )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.manual_seed(args.seed + 1)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 1)

        for _ in range(args.warmup_steps):
            training_step(
                model,
                optimizer,
                idx,
                targets,
                starts,
                best_of_k,
                args,
            )
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        losses = []
        for _ in range(args.steps):
            losses.append(
                training_step(
                    model,
                    optimizer,
                    idx,
                    targets,
                    starts,
                    best_of_k,
                    args,
                )
            )
        synchronize(device)
        elapsed = time.perf_counter() - start
        step_seconds = elapsed / args.steps
        results.append({
            "xm_best_of_k": best_of_k,
            "mean_loss": sum(losses) / len(losses),
            "step_seconds": step_seconds,
            "data_tokens_per_second": (
                args.batch_size * args.sequence_length / step_seconds
            ),
            "peak_memory_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None
            ),
            "theoretical_training_flop_factor": (
                1.0 if best_of_k == 1 else (best_of_k + 3) / 3
            ),
        })
    return {
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "memory_saving_xm": True,
        "results": results,
    }


def build_model(args, vocab_size, best_of_k):
    return Decifer(DeciferConfig(
        tokenizer="minicif_v2",
        vocab_size=vocab_size,
        block_size=args.sequence_length,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embedding_dim,
        dropout=args.dropout,
        record_aligned_attention=True,
        xm_best_of_k=best_of_k,
    ))


def synthetic_batch(batch_size, sequence_length, vocab_size, start_id, device):
    tokens = torch.randint(
        1,
        vocab_size,
        (batch_size, sequence_length + 1),
        device=device,
    )
    tokens[:, 0] = start_id
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def training_step(model, optimizer, idx, targets, starts, best_of_k, args):
    optimizer.zero_grad(set_to_none=True)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    context = (
        nullcontext()
        if dtype == torch.float32
        else torch.autocast(device_type=idx.device.type, dtype=dtype)
    )
    with context:
        if best_of_k == 1:
            _, loss = model(idx, targets=targets, start_indices_batch=starts)
        else:
            _, loss = xm_best_of_k_forward(
                model,
                idx,
                None,
                targets,
                starts,
                best_of_k,
            )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if any(best_of_k < 1 for best_of_k in args.k):
        parser.error("all K values must be >= 1")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be >= 1")

    result = benchmark(args)
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()
