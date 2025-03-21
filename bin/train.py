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

from dataclasses import dataclass, field
from contextlib import nullcontext
from tqdm.auto import tqdm

from omegaconf import OmegaConf

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import discrete_to_continuous_xrd
from decifer.decifer_dataset import DeciferDataset
    
# Tokenizer, get start, padding and newline IDs
TOKENIZER = Tokenizer()
VOCAB_SIZE = TOKENIZER.vocab_size
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]

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

@dataclass
class TrainConfig:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # Path to the dataset hdf5 files
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters
    cond_size: int = 1000
    accumulative_pbar: bool = False
    num_workers_dataloader: int = 0 # Default; single process

    # deCIFer model
    block_size: int = 1024
    vocab_size: int = 372 # Excluding conditioning token
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
    boundary_masking: bool = True

    # PXRD embedder
    condition: bool = False
    condition_embedder_hidden_layers: List[int] = field(default_factory=lambda: [512])

    # Augmentation at training time
    qmin: float = 0.0
    qmax: float = 10.0
    qstep: float = 0.01
    wavelength: str = "CuKa"
    fwhm_range_min: float = 0.001 
    fwhm_range_max: float = 0.05
    eta_range_min: float = 0.5
    eta_range_max: float = 0.5
    noise_range_min: float = 0.001
    noise_range_max: float = 0.05
    intensity_scale_range_min: float = 1.0
    intensity_scale_range_max: float = 1.0
    mask_prob: float = 0.0

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 50_000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster (Not supported for deCIFer currently)
    validate: bool = False  # whether to evaluate the model using the validation set
    seed: int = 1337

    # Early stopping
    early_stopping_patience: int = 50

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to .yaml config file")
    args = parser.parse_args()

    C = OmegaConf.structured(TrainConfig())

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Parse yaml to namespace and merge (DictConfig)
        yaml_dictconfig = OmegaConf.create(yaml_config)
        C = OmegaConf.merge(C, yaml_dictconfig)
    
    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")
    
    print("Using configuration:", flush=True)
    print(OmegaConf.to_yaml(C))
    
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
    train_dataset = DeciferDataset(os.path.join(C.dataset, "serialized/train.h5"), dataset_fields)
    val_dataset = DeciferDataset(os.path.join(C.dataset, "serialized/val.h5"), dataset_fields)
    test_dataset = DeciferDataset(os.path.join(C.dataset, "serialized/test.h5"), dataset_fields)
        
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

if __name__ == "__main__":

    # Parse configuration
    C = parse_config()
    
    # Set seed
    if C.seed is not None: torch.manual_seed(C.seed)
    
    # Setup ctx, note: float16 data type will automatically use a GradScaler
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if C.device == "cpu" else torch.cuda.amp.autocast(dtype=ptdtype)

    # Setup datasets
    dataloaders = setup_datasets(C)

    # Augmentation kwargs
    augmentation_kwargs = {
        'qmin': C.qmin,
        'qmax': C.qmax,
        'qstep': C.qstep,
        # 'wavelength': C.wavelength,
        'fwhm_range': (C.fwhm_range_min, C.fwhm_range_max),
        'eta_range': (C.eta_range_min, C.eta_range_max),
        'noise_range': (C.noise_range_min, C.noise_range_max),
        'intensity_scale_range': (C.intensity_scale_range_min, C.intensity_scale_range_max),
        'mask_prob': C.mask_prob,
        # 'size': len(np.arange(C.qmin, C.qmax, C.qstep))
    }

    # Initialize training metrics
    training_metrics = {
        'iteration_number': 0,
        'patience_counter': 0,
        'best_val_loss': float('inf'),
        'train_losses': [],
        'val_losses': [],
        'epochs': [],
    }

    # Set model arguments
    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head, 
        n_embd=C.n_embd, 
        block_size=C.block_size,
        condition_size=len(np.arange(C.qmin, C.qmax, C.qstep)),
        bias=C.bias,
        vocab_size=C.vocab_size,
        dropout=C.dropout,
        condition=C.condition,
        boundary_masking=C.boundary_masking,
        condition_embedder_hidden_layers = C.condition_embedder_hidden_layers,
    )

    if C.init_from == "scratch":
        print("Initializing a new model from scratch...", flush=True)
        model = Decifer(DeciferConfig(**model_args))
    
        checkpoint = {
            'model_args': model_args,
            'training_metrics': training_metrics,
            'best_model_state': None,
            'best_optimizer_state': None,
            "local_iteration_number": 0,
            'config': dict(C),
        }

    elif C.init_from == "resume":
        print(f"Resuming training from {C.out_dir}...", flush=True)

        # Find checkpoint
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]

        # Force these config attributes to be equal otherwise we can't even resume training
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        # Init model and load state dict
        model = Decifer(DeciferConfig(**model_args))
        state_dict = checkpoint['current_model']
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        # Update checkpoint
        for key in ['train_losses', 'val_losses', 'epochs']:
            if key in checkpoint['training_metrics']:
                training_metrics[key] = checkpoint['training_metrics'][key]
                print(f"Loaded {key}.")
            else:
                print(f"Could not find {key}, creating empty list")

        training_metrics['iteration_number'] = checkpoint["training_metrics"]["iteration_number"]
        training_metrics['best_val_loss'] = checkpoint["training_metrics"]["best_val_loss"]
    else:
        raise Exception(f"[init_from] '{C.init_from}' not recognized")

    # Send model to device
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    # Initialize Optimizer
    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        optimizer.load_state_dict(checkpoint["current_optimizer"])

    # Compile model (pytorch 2.0) if specified
    if C.compile:
        print("Compiling the model (takes a ~minute)...", flush=True)
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
    
    # Initialize a dictionary to keep data iterators per split
    data_iters = {}

    #@profile
    def get_batch(split):

        # Retrieve the dataloader and initialize the iterator
        dataloader = dataloaders[split]
        if split not in data_iters:
            data_iters[split] = iter(dataloader)
        data_iter = data_iters[split]

        # Initialize lists to store packed sequences and start indices
        start_indices_list = []
        cond_list = []

        # Collect sequences until we have enough to fill the batch
        total_sequences = []
        while len(total_sequences) < C.batch_size:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data_iters[split] = data_iter
                batch = next(data_iter)

            # Fetch sequences and remove padding tokens
            sequences = batch['cif_tokens']
            sequences = [torch.cat([seq[seq != PADDING_ID], torch.tensor([NEWLINE_ID, NEWLINE_ID], dtype=torch.long)]) for seq in sequences]
            total_sequences.extend(sequences)

            # Fetch conditioning and augment to cont signals
            if C.condition:
                cond_list.extend(discrete_to_continuous_xrd(batch['xrd.q'], batch['xrd.iq'], **augmentation_kwargs)['iq'])

        # Now pack sequences into batches without loops
        # Concatenate all sequences into one long tensor
        all_tokens = torch.cat(total_sequences)

        # Compute the lengths of sequences
        seq_lengths = torch.tensor([len(seq) for seq in total_sequences])

        # Compute cumulative lengths to find sequence boundaries
        seq_cum_lengths = torch.cumsum(seq_lengths, dim=0)

        # Calculate how many full blocks we can get from the concatenated tokens
        num_full_blocks = all_tokens.size(0) // C.block_size
        num_batches = min(C.batch_size, num_full_blocks)

        # Truncate the tokens to fit into an integer number of blocks
        total_tokens = all_tokens[:num_batches * C.block_size]

        # Reshape the tokens into (num_batches, block_size)
        total_tokens = total_tokens.view(num_batches, C.block_size)

        # Create input (X) and target (Y) sequences
        X_batch = total_tokens[:, :-1]
        Y_batch = total_tokens[:, 1:]

        # Find start indices within each batch
        start_token_mask = X_batch == START_ID
        start_indices = start_token_mask.nonzero(as_tuple=False)

        # Organize start indices per batch item
        start_indices_list = []
        for i in range(num_batches):
            indices = start_indices[start_indices[:, 0] == i][:, 1]
            start_indices_list.append(indices)

        # Handle conditioning data if required
        # Collect conditioning data corresponding to sequences included
        cond_batch = None
        if C.condition:
            index = torch.searchsorted(seq_cum_lengths, num_batches * C.block_size) + 1
            cond_list = cond_list[:index]
            cond_batch = torch.stack(cond_list)
        
        # Send to device (CUDA/CPU)
        if C.device == "cuda":
            X_batch = X_batch.pin_memory().to(C.device, non_blocking=True)
            Y_batch = Y_batch.pin_memory().to(C.device, non_blocking=True)
            if cond_batch is not None:
                cond_batch = cond_batch.pin_memory().to(C.device, non_blocking=True)
        else:
            X_batch = X_batch.pin_memory().to(C.device)
            Y_batch = Y_batch.pin_memory().to(C.device)
            if cond_batch is not None:
                cond_batch = cond_batch.pin_memory().to(C.device)

        # Return the batch data and start indices
        return X_batch, Y_batch, cond_batch, start_indices_list

    # helps estimate an arbitrarily accurate loss over either split using many batches
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

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > C.lr_decay_iters:
            return C.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    # training loop
    X, Y, cond, start_indices = get_batch("train")
    t0 = time.time()
    local_iteration_number = 0  # number of iterations in the lifetime of this process
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(training_metrics['iteration_number']) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if training_metrics['iteration_number'] % C.eval_interval == 0:
            if C.validate:

                # Esimate loss
                losses = estimate_loss()

                # Update metrics
                training_metrics['train_losses'].append(losses['train'])
                training_metrics['val_losses'].append(losses['val'])
                training_metrics['epochs'].append(training_metrics['iteration_number'])
                print(f"step {training_metrics['iteration_number']}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

                # Check if new best model is found; if so, save, else patience score recal
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

                    # Update checkpoint
                    checkpoint.update({
                        "local_iteration_number": local_iteration_number,
                        'training_metrics': training_metrics,
                        'current_model': model.state_dict(),
                        "current_optimizer": optimizer.state_dict(),
                    })

                    print(f"saving checkpoint to {C.out_dir}...", flush=True)
                    torch.save(checkpoint, os.path.join(C.out_dir, "ckpt.pt"))

                if training_metrics['patience_counter'] >= C.early_stopping_patience:
                    print(f"Early stopping triggered after {training_metrics['iteration_number']} iterations")
                    break

            else:
                training_metrics['best_val_loss'] = 0.

        if training_metrics['iteration_number'] == 0 and C.eval_only:
            break

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        small_step_pbar = tqdm(desc='Accumulating losses...', total=C.gradient_accumulation_steps, leave=False, disable=not C.accumulative_pbar)
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, cond, Y, start_indices)
                
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, cond, start_indices = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            small_step_pbar.update(1)

        small_step_pbar.close()
        # clip the gradient
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if training_metrics['iteration_number'] % C.log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            print(f"iter {training_metrics['iteration_number']}: loss {lossf:.4f}, time {dt * 1000:.2f}ms", flush=True)
        training_metrics['iteration_number'] += 1
        local_iteration_number += 1

        # termination conditions
        if training_metrics['iteration_number'] > C.max_iters:
            break
