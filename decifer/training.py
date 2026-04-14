#!/usr/bin/env python3

"""
Adapted from:
nanoGPT: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
CrystaLLM: https://github.com/lantunes/CrystaLLM/blob/main/bin/train.py
"""
import os
import copy
import math
import time
import yaml
import random

from typing import List
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import BatchSampler

from torch.nn.utils.rnn import pad_sequence

from contextlib import nullcontext
from tqdm.auto import tqdm

from decifer.config import TrainWorkflowConfig, dataclass_to_dict, load_dataclass_config
from decifer.datasets import load_decifer_dataset, resolve_dataset_splits
from decifer.decifer_model import Decifer, DeciferConfig
from decifer.io import create_run_layout
from decifer.tokenizer import Tokenizer
from decifer.utility import discrete_to_continuous_xrd
from decifer.decifer_dataset import DeciferDataset
    
# Tokenizer, get start, padding and newline IDs
TOKENIZER = Tokenizer()
VOCAB_SIZE = TOKENIZER.vocab_size
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]

TrainConfig = TrainWorkflowConfig

class RandomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Each time __iter__ is called, radomize the batch indices
        batch_indices = list(self.sampler)
        random.shuffle(batch_indices)

        # Return batches of size batch_Size
        for i in range(0, len(batch_indices), self.batch_size):
            yield batch_indices[i:i + self.batch_size]

def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to .yaml config file")
    args = parser.parse_args(argv)

    C = load_dataclass_config(TrainConfig, config_path=args.config)
    
    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")
    
    print("Using configuration:", flush=True)
    print(yaml.safe_dump(dataclass_to_dict(C), sort_keys=False))
    
    # Creating output
    print(f"Creating {C.out_dir}...", flush=True)
    os.makedirs(C.out_dir, exist_ok=True)

    # Get metadata (vocab size)
    # metadata_path = os.path.join(C.dataset, "metadata.json")
    # with open(metadata_path, "r") as f:
    #     metadata = json.load(f)
    # try:
    #     print(metadata)
    #     C.vocab_size = metadata["vocab_size"]
    #     print(f"Found vocab_size = {C.vocab_size} in {metadata_path}", flush=True)
    # except:
    #     print(f"No metadata for vocab_size found, defaulting to {C.vocab_size}...")
    C.vocab_size = VOCAB_SIZE

    return C

def setup_datasets(C):
    
    # Custom collate function
    def collate_fn(batch):
        # batch is a list of dictionaries
        batch_data = {}
        for key in batch[0].keys():
            field_data = [item[key] for item in batch]
            # Pad the sequences to the maximum length in the batch
            if "xrd" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=0.0)
                batch_data[key] = padded_seqs
            elif "cif" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=PADDING_ID)
                batch_data[key] = padded_seqs
            else:
                batch_data[key] = field_data  # Leave 

        return batch_data
    
    # Collect relevant data
    dataset_fields = ["cif_tokens", "xrd.q", "xrd.iq"]

    # Initialise datasets/loaders 
    split_paths = resolve_dataset_splits(C.dataset)
    train_dataset = load_decifer_dataset(split_paths["train"], dataset_fields, dataset_cls=DeciferDataset)
    val_dataset = load_decifer_dataset(split_paths["val"], dataset_fields, dataset_cls=DeciferDataset)
    test_dataset = load_decifer_dataset(split_paths["test"], dataset_fields, dataset_cls=DeciferDataset)
        
    # Random batching sampler, train
    train_sampler = SubsetRandomSampler(range(len(train_dataset)))
    train_batch_sampler = RandomBatchSampler(train_sampler, batch_size=C.batch_size, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=C.num_workers_dataloader, collate_fn=collate_fn)
    
    # Random batching sampler, val
    val_sampler = SubsetRandomSampler(range(len(val_dataset)))
    val_batch_sampler = RandomBatchSampler(val_sampler, batch_size=C.batch_size, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=C.num_workers_dataloader, collate_fn=collate_fn)
    
    # Random batching sampler, test
    test_sampler = SubsetRandomSampler(range(len(test_dataset)))
    test_batch_sampler = RandomBatchSampler(test_sampler, batch_size=C.batch_size, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=C.num_workers_dataloader, collate_fn=collate_fn)

    # Combine loaders for easy access
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

    return dataloaders


def resolve_run_dir(config):
    return os.path.abspath(config.out_dir)

def build_augmentation_kwargs(config):
    return {
        'qmin': config.qmin,
        'qmax': config.qmax,
        'qstep': config.qstep,
        'fwhm_range': (config.fwhm_range_min, config.fwhm_range_max),
        'eta_range': (config.eta_range_min, config.eta_range_max),
        'noise_range': (config.noise_range_min, config.noise_range_max),
        'intensity_scale_range': (config.intensity_scale_range_min, config.intensity_scale_range_max),
        'mask_prob': config.mask_prob,
    }


def initialize_training_metrics():
    return {
        'iteration_number': 0,
        'patience_counter': 0,
        'best_val_loss': float('inf'),
        'train_losses': [],
        'val_losses': [],
        'epochs': [],
    }


def snapshot_training_metrics(training_metrics, latest_train_loss=None, learning_rate=None, status=None):
    payload = dict(training_metrics)
    if latest_train_loss is not None:
        payload["latest_train_loss"] = latest_train_loss
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate
    if status is not None:
        payload["status"] = status
    return payload


def move_batch_tensor(tensor, device):
    if tensor is None:
        return None
    device = str(device)
    if device.startswith("cuda"):
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def build_training_block_batch(sequences, cond_sequences, block_size, batch_size, condition):
    if not sequences:
        return None

    all_tokens = torch.cat(sequences)
    num_full_blocks = all_tokens.size(0) // block_size
    if num_full_blocks == 0:
        return None

    total_tokens = all_tokens[:num_full_blocks * block_size].view(num_full_blocks, block_size)
    x_candidates = total_tokens[:, :-1]
    y_candidates = total_tokens[:, 1:]

    start_token_mask = x_candidates == START_ID
    valid_block_indices = torch.nonzero(start_token_mask.any(dim=1), as_tuple=False).flatten()
    if valid_block_indices.numel() == 0:
        return None

    num_batches = min(batch_size, int(valid_block_indices.numel()))
    selected_indices = valid_block_indices[:num_batches]
    x_batch = x_candidates[selected_indices]
    y_batch = y_candidates[selected_indices]

    start_indices_list = []
    for block_index in selected_indices.tolist():
        start_indices = torch.nonzero(start_token_mask[block_index], as_tuple=False).flatten()
        start_indices_list.append(start_indices)

    cond_batch = None
    if condition:
        total_insertions = sum(int(indices.numel()) for indices in start_indices_list)
        if len(cond_sequences) < total_insertions:
            return None
        cond_batch = torch.stack(cond_sequences[:total_insertions])

    return x_batch, y_batch, cond_batch, start_indices_list


def build_model_args(config):
    return dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        condition_size=len(np.arange(config.qmin, config.qmax, config.qstep)),
        bias=config.bias,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        condition=config.condition,
        boundary_masking=config.boundary_masking,
        condition_embedder_hidden_layers=config.condition_embedder_hidden_layers,
    )


def initialize_model_and_checkpoint(config, model_args, training_metrics):
    if config.init_from == "scratch":
        print("Initializing a new model from scratch...", flush=True)
        model = Decifer(DeciferConfig(**model_args))

        checkpoint = {
            'model_args': model_args,
            'training_metrics': training_metrics,
            'best_model_state': None,
            'best_optimizer_state': None,
            "local_iteration_number": 0,
            'config': dataclass_to_dict(config),
        }
        return model, checkpoint

    if config.init_from == "resume":
        print(f"Resuming training from {config.out_dir}...", flush=True)

        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        checkpoint_model_args = checkpoint["model_args"]

        for key in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[key] = checkpoint_model_args[key]

        model = Decifer(DeciferConfig(**model_args))
        state_dict = checkpoint['current_model']
        unwanted_prefix = "_orig_mod."
        for key, value in list(state_dict.items()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
        model.load_state_dict(state_dict)

        for key in ['train_losses', 'val_losses', 'epochs']:
            if key in checkpoint['training_metrics']:
                training_metrics[key] = checkpoint['training_metrics'][key]
                print(f"Loaded {key}.")
            else:
                print(f"Could not find {key}, creating empty list")

        training_metrics['iteration_number'] = checkpoint["training_metrics"]["iteration_number"]
        training_metrics['best_val_loss'] = checkpoint["training_metrics"]["best_val_loss"]
        return model, checkpoint

    raise Exception(f"[init_from] '{config.init_from}' not recognized")


def run_training(C):
    layout = create_run_layout(
        resolve_run_dir(C),
        "train",
        C,
        metadata={
            "dataset": os.path.abspath(C.dataset),
            "resolved_splits": resolve_dataset_splits(C.dataset),
            "out_dir": os.path.abspath(C.out_dir),
        },
    )

    if C.seed is not None: torch.manual_seed(C.seed)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if C.device == "cpu" else torch.cuda.amp.autocast(dtype=ptdtype)

    dataloaders = setup_datasets(C)
    augmentation_kwargs = build_augmentation_kwargs(C)
    training_metrics = initialize_training_metrics()
    model_args = build_model_args(C)
    model, checkpoint = initialize_model_and_checkpoint(C, model_args, training_metrics)

    model.to(C.device)

    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        optimizer.load_state_dict(checkpoint["current_optimizer"])

    if C.compile:
        print("Compiling the model (takes a ~minute)...", flush=True)
        model = torch.compile(model)  # requires PyTorch 2.0

    data_iters = {}

    def get_batch(split):
        dataloader = dataloaders[split]
        if split not in data_iters:
            data_iters[split] = iter(dataloader)
        data_iter = data_iters[split]

        cond_list = []
        total_sequences = []
        wrapped_dataloader = 0

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data_iters[split] = data_iter
                wrapped_dataloader += 1
                if wrapped_dataloader > 1:
                    raise ValueError(
                        f"Could not assemble a valid {split} batch with block_size={C.block_size}. "
                        "The dataset may be too short for the configured block size."
                    )
                batch = next(data_iter)

            sequences = batch['cif_tokens']
            sequences = [torch.cat([seq[seq != PADDING_ID], torch.tensor([NEWLINE_ID, NEWLINE_ID], dtype=torch.long)]) for seq in sequences]
            total_sequences.extend(sequences)

            if C.condition:
                cond_list.extend(discrete_to_continuous_xrd(batch['xrd.q'], batch['xrd.iq'], **augmentation_kwargs)['iq'])

            batch_tensors = build_training_block_batch(
                sequences=total_sequences,
                cond_sequences=cond_list,
                block_size=C.block_size,
                batch_size=C.batch_size,
                condition=C.condition,
            )
            if batch_tensors is None:
                continue

            X_batch, Y_batch, cond_batch, start_indices_list = batch_tensors
            X_batch = move_batch_tensor(X_batch, C.device)
            Y_batch = move_batch_tensor(Y_batch, C.device)
            cond_batch = move_batch_tensor(cond_batch, C.device)
            return X_batch, Y_batch, cond_batch, start_indices_list

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, cond, start_indices = get_batch(split)
                with ctx:
                    _, loss = model(X, cond, Y, start_indices)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        if it > C.lr_decay_iters:
            return C.min_lr
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    X, Y, cond, start_indices = get_batch("train")
    t0 = time.time()
    local_iteration_number = 0
    while True:
        lr = get_lr(training_metrics['iteration_number']) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if training_metrics['iteration_number'] % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss()
                training_metrics['train_losses'].append(losses['train'])
                training_metrics['val_losses'].append(losses['val'])
                training_metrics['epochs'].append(training_metrics['iteration_number'])
                print(f"step {training_metrics['iteration_number']}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

                if losses["val"] > training_metrics['best_val_loss'] and local_iteration_number != 0:
                    training_metrics['patience_counter'] += 1
                    print("Patience score increasing to:", training_metrics['patience_counter'])
                else:
                    training_metrics['best_val_loss'] = losses['val']
                    checkpoint['best_model_state'] = copy.deepcopy(model.state_dict())
                    checkpoint['best_optimizer_state'] = copy.deepcopy(optimizer.state_dict())
                    if training_metrics['patience_counter'] > 0:
                        print("Patience score resetting.")
                        training_metrics['patience_counter'] = 0

                if training_metrics['iteration_number'] > 0:
                    checkpoint.update({
                        "local_iteration_number": local_iteration_number,
                        'training_metrics': training_metrics,
                        'current_model': model.state_dict(),
                        "current_optimizer": optimizer.state_dict(),
                    })

                    print(f"saving checkpoint to {C.out_dir}...", flush=True)
                    torch.save(checkpoint, os.path.join(C.out_dir, "ckpt.pt"))
                    layout.write_metadata({"checkpoint_path": os.path.join(C.out_dir, "ckpt.pt")})
                    layout.write_metrics(
                        snapshot_training_metrics(training_metrics, learning_rate=lr, status="running")
                    )

                if training_metrics['patience_counter'] >= C.early_stopping_patience:
                    print(f"Early stopping triggered after {training_metrics['iteration_number']} iterations")
                    break

            else:
                training_metrics['best_val_loss'] = 0.

        if training_metrics['iteration_number'] == 0 and C.eval_only:
            break

        small_step_pbar = tqdm(desc='Accumulating losses...', total=C.gradient_accumulation_steps, leave=False, disable=not C.accumulative_pbar)
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, cond, Y, start_indices)
                loss = loss / C.gradient_accumulation_steps

            X, Y, cond, start_indices = get_batch("train")
            scaler.scale(loss).backward()
            small_step_pbar.update(1)

        small_step_pbar.close()
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if training_metrics['iteration_number'] % C.log_interval == 0:
            lossf = loss.item() * C.gradient_accumulation_steps
            print(f"iter {training_metrics['iteration_number']}: loss {lossf:.4f}, time {dt * 1000:.2f}ms", flush=True)
            layout.write_metrics(
                snapshot_training_metrics(
                    training_metrics,
                    latest_train_loss=lossf,
                    learning_rate=lr,
                    status="running",
                )
            )
        training_metrics['iteration_number'] += 1
        local_iteration_number += 1

        if training_metrics['iteration_number'] > C.max_iters:
            break

    layout.write_metrics(snapshot_training_metrics(training_metrics, learning_rate=lr, status="completed"))
    return {
        "layout": layout,
        "training_metrics": training_metrics,
        "checkpoint": checkpoint,
    }


def main(argv=None):
    return run_training(parse_config(argv))


if __name__ == "__main__":
    main()
