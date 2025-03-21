#!/usr/bin/env python3

"""
Inspired by and adapted from:
nanoGPT: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py (MIT License)
CrystaLLM: https://github.com/lantunes/CrystaLLM/blob/main/crystallm/_model.py
"""

import math, random, re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict
from h5py._hl.files import sys
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from decifer.tokenizer import Tokenizer
TOKENIZER = Tokenizer()
NEWLINE_ID = TOKENIZER.token_to_id["\n"]
PADDING_ID = TOKENIZER.padding_id
TOKENIZE = TOKENIZER.tokenize_cif
ENCODE = TOKENIZER.encode
DECODE = TOKENIZER.decode

@dataclass
class DeciferConfig:
    block_size: int = 1024
    vocab_size: int = 372
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
    condition: bool = False
    boundary_masking: bool = True
    condition_size: int = 1000
    condition_embedder_hidden_layers: List[int] = field(default_factory=lambda: [512])
    plot_attention: bool = False
    print_parameter_info: bool = False

class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool):
        """
        Initialize the LayerNorm module.

        :param ndim: dimensionality of the input tensor
        :param bias: whether to add a learnable bias to the output
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        # Masked attention
        self.masked_bias = torch.tensor(-1e4)

    def forward(self, x: torch.Tensor, attention_bias: Optional[torch.Tensor] = None, return_attn: bool = False) -> torch.Tensor:
        """
        Applies causal self-attention to the given tensor,
        with a mask to prevent attention to future positions.

        :param x: tensor of shape (batch size, sequence length, embedding dimension)
        :returns: result of applying the causal self-attention operation
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        #causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and not return_attn:
            if attention_bias is not None:
                # Expand attention_bias to match the number of heads
                attention_bias = attention_bias.unsqueeze(1) # Shape (B, 1, T, T)
                attention_bias = attention_bias.expand(B, self.n_head, T, T)  # Expand to (B, n_head, T, T)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_bias, dropout_p=self.dropout
                )
            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True,
                )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_bias is not None:
                attention_bias = attention_bias.unsqueeze(1) # Shap (B, 1, T, T)
                att = att + attention_bias
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        if not return_attn:
            return y
        else:
            return y, att

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function, as used in the Google BERT and
    OpenAI GPT models. See: "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415

    :param x: the tensor to which the GELU activation function will be applied
    :returns: the result tensor after applying the GELU activation function,
              possessing the same shape as the input tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_bias: Optional[torch.Tensor] = None, return_attn: bool = False, print_parameters: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass for the Transformer Block module. A Block module includes causal self-attention,
        layer normalization, and MLP, and residual connections.

        :param x: input to the transformer block
        :returns: output of the transformer block, with the same shape as in the input
        """
        if not return_attn:
            x = x + self.attn(self.ln_1(x), attention_bias)
            x = x + self.mlp(self.ln_2(x))
            return x
        else:
            y, att = self.attn(self.ln_1(x), attention_bias, return_attn=return_attn)
            x = x + y
            x = x + self.mlp(self.ln_2(x))
            return x, att

class Decifer(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.condition_size is not None

        self.config = config

        self.tokenizer = Tokenizer()

        self.attn_scores = None

        # Condtional embedding: either using straight MLP or direct CL encoding
        if config.condition:
            # MLP to embedding size
            cond_embedding = nn.Sequential(
                nn.Linear(config.condition_size, config.condition_embedder_hidden_layers[0]),
                nn.ReLU(),
                *[
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
                    for hidden_size in config.condition_embedder_hidden_layers[:-1]
                ],
                nn.Linear(config.condition_embedder_hidden_layers[-1], config.n_embd)
            )
        else:
            cond_embedding = nn.Identity() # nn's version of None

        self.transformer = nn.ModuleDict(dict(
            cond_embedding=cond_embedding,
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        if self.config.print_parameter_info:
            print("number of total non-trainable parameters: %.2fM" % (self.get_num_params()/1e6,))
            print("number of total trainable parameters: %.2fM" % (self.get_num_params(trainable=True)/1e6,))
            if self.config.condition:
                print("number of total conditioning MLP parameters: %.2fM" % (self.get_num_params(return_cond_mlp=True)/1e6,))

    def get_num_params(self, non_embedding: bool = True, trainable: bool = False, return_cond_mlp: bool = False) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :param trainable: whether to return trainable parameters that requires_grad
        :param return_cond_mlp: return only conditional mlp.
        :returns: the number of parameters in the model
        """
        if return_cond_mlp:
            assert self.config.condition == True
            parameters = self.transformer.cond_embedding.parameters()
        else:
            parameters = self.parameters()

        if not trainable:
            n_params = sum(p.numel() for p in parameters)
            if non_embedding and not return_cond_mlp:
                n_params -= self.transformer.wpe.weight.numel()
        else:
            n_params = sum(p.numel() for p in parameters if p.requires_grad)

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        start_indices_batch: List[List[int]] = [[0]],
        custom_cond_emb: Optional[torch.Tensor] = None,
    ):

        device = idx.device
        b, t = idx.size()
        ptdtype = self.transformer.wte.weight.dtype

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Original positions
        positions = torch.arange(t, device=device).unsqueeze(0).expand(b, t)  # shape (b, t)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)  # shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(positions)  # shape (b, t, n_embd)

        # Initialize variables
        attention_bias = None
            
        # Convert start indices to list of tensor
        start_indices_tensors = [torch.tensor(s, dtype=torch.long, device=device) for s in start_indices_batch]

        if self.config.condition:

            # Convert start indices to a list of tensors
            start_indices_tensors = [torch.tensor(s, dtype=torch.long, device=device) for s in start_indices_batch]
            # Pad sequences to the same length
            start_indices_padded = nn.utils.rnn.pad_sequence(start_indices_tensors, batch_first=True, padding_value=t) # Padded with t
            # Computer the number of inserts per sequence
            num_inserts_per_seq = torch.tensor([len(s) for s in start_indices_batch], dtype=torch.long, device=device)
            max_num_inserts = start_indices_padded.size(1)
            
            # Compute new sequence lengths
            new_lengths = t + num_inserts_per_seq
            max_new_t = new_lengths.max().item()
            
            # Compute valid insertion mask and positions
            valid_insert_mask = start_indices_padded < t
            insertion_offsets = torch.cumsum(valid_insert_mask.long(), dim=1) - 1
            adjusted_insert_positions = start_indices_padded + insertion_offsets
            adjusted_insert_positions = adjusted_insert_positions[valid_insert_mask]

            batch_indices = torch.arange(b, device=device).unsqueeze(1).expand(-1, max_num_inserts)  # shape (b, max_num_inserts)
            batch_indices = batch_indices[valid_insert_mask]
            insert_mask = torch.zeros((int(b), int(max_new_t)), dtype=torch.bool, device=device)
            insert_mask[batch_indices, adjusted_insert_positions] = True
            
            # Build index map for gathering original embeddings
            cumsum_insert_mask = torch.cumsum(insert_mask, dim=1)
            positions_in_new_seq = torch.arange(max_new_t, device=device).unsqueeze(0).expand(b, -1)
            orig_positions = positions_in_new_seq - cumsum_insert_mask
            valid_positions = (orig_positions >= 0) & (orig_positions < t) & (~insert_mask)
            index_map = torch.full((int(b), int(max_new_t)), -1, dtype=torch.long, device=device)  # -1 indicates conditioning
            index_map[valid_positions] = orig_positions[valid_positions]
            
            # Create masks for cond_emb and tok_emb positions
            tok_emb_mask = (index_map >= 0)    # Shape: (b, max_new_t)

            # Compute cond_emb
            if custom_cond_emb is None:
                cond_emb = self.transformer.cond_embedding(cond_vec).to(dtype=ptdtype)  # Shape: (total_insertions, n_embd)
            else:
                cond_emb = custom_cond_emb

            # Prepare indices for gathering tok_emb and pos_emb
            gather_indices = index_map.clamp(min=0)
            tok_emb_new = torch.zeros((int(b), int(max_new_t), int(self.config.n_embd)), device=device, dtype=ptdtype)
            pos_emb_new = torch.zeros_like(tok_emb_new)

            # Find valid token positions
            batch_idx, seq_pos = torch.nonzero(tok_emb_mask, as_tuple=True)
            orig_pos = gather_indices[batch_idx, seq_pos]

            # Gather the embeddings and remap to the new embedding tensors
            tok_emb_values = tok_emb[batch_idx, orig_pos, :]
            pos_emb_values = pos_emb[batch_idx, orig_pos, :]
            tok_emb_new[batch_idx, seq_pos, :] = tok_emb_values
            pos_emb_new[batch_idx, seq_pos, :] = pos_emb_values

            cond_indices = torch.argwhere(insert_mask)
            tok_emb_new[cond_indices[:,0], cond_indices[:,1]] = cond_emb[:sum(num_inserts_per_seq)] # Truncated in case of batch truncation

            # Update variables
            tok_emb = tok_emb_new
            pos_emb = pos_emb_new
            positions = positions_in_new_seq
            t = max_new_t

            # Construct attention bias
            start_mask = insert_mask.long()
            group_ids = torch.cumsum(start_mask, dim=1)
            causal_mask = positions.unsqueeze(2) >= positions.unsqueeze(1)
            group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
            attention_mask = causal_mask & group_mask
            attention_bias = torch.where(attention_mask, torch.zeros(1, dtype=ptdtype, device=device), 
                                         torch.full((1,), float('-inf'), dtype=ptdtype, device=device))
            # attention_bias is (B, T, T)

            # Adjust targets if provided
            if targets is not None:
                # Gather indices where index_map >= 0
                valid_positions = index_map >= 0
                batch_idx, seq_pos = torch.nonzero(valid_positions, as_tuple=True)
                orig_pos = index_map[batch_idx, seq_pos]

                # Gather idx and targets
                idx_new = torch.full((int(b), int(max_new_t)), Tokenizer().padding_id, dtype=idx.dtype, device=device)
                targets_new = torch.full((int(b), int(max_new_t)), -1, dtype=targets.dtype, device=device)

                idx_new[batch_idx, seq_pos] = idx[batch_idx, orig_pos]
                targets_new[batch_idx, seq_pos] = targets[batch_idx, orig_pos]

                # Update idx and targets
                idx = idx_new
                targets = targets_new

        else:
            if self.config.boundary_masking:

                start_mask = torch.zeros((b, t), dtype=torch.long, device=device)
                for i, start_indices in enumerate(start_indices_batch):
                    valid_indices = torch.tensor(start_indices, device=device)
                    valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < t)]
                    start_mask[i, valid_indices] = 1
                group_ids = torch.cumsum(start_mask, dim=1)
                
                causal_mask = positions.unsqueeze(2) >= positions.unsqueeze(1)
                group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
                attention_mask = causal_mask & group_mask
                
                attention_bias = torch.where(attention_mask, torch.zeros(1, dtype=ptdtype, device=device),
                                             torch.full((1,), float('-inf'), dtype=ptdtype, device=device))
                # attention_bias is (B, T, T)

        # Combine token and position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward pass through transformer blocks
        self.attn_scores = []
        for i, block in enumerate(self.transformer.h):
            x, att = block(x, attention_bias=attention_bias, return_attn=True)
            self.attn_scores.append(att.detach().cpu().mean(dim=1))

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode
            logits = self.lm_head(x[:, [-1], :])  # only the last token
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: "transformer.wte.weight" and "lm_head.weight" are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, keyed by "transformer.wte.weight", below.
        # so let's manually remove "lm_head.weight" from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, cond_vec=None, start_indices_batch=None, temperature=1.0, top_k=None, disable_pbar=False, custom_cond_emb=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        prev_id = None
        generation_pbar = tqdm(total=max_new_tokens, desc='Generating sequence', leave=False, disable=disable_pbar)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, cond_vec=cond_vec, start_indices_batch = start_indices_batch, custom_cond_emb=custom_cond_emb)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # a sequence of two newlines indicates the end of a CIF file
            if prev_id is not None and prev_id == NEWLINE_ID and idx_next.item() == NEWLINE_ID:
                break
            # as soon as <pad> is hit -> end of cif
            if idx_next.item() == PADDING_ID:
                idx = idx[:,:-1]
                break
            prev_id = idx_next.item()
            generation_pbar.update(1)
        generation_pbar.close()
        
        return idx

    @torch.no_grad()
    def generate_batched_reps(self, idx, max_new_tokens, cond_vec=None, start_indices_batch=None, temperature=1.0, top_k=None, disable_pbar=False, custom_cond_emb = None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (batch_size, seq_len)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        batch_size = idx.size(0)
        device = idx.device

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_id = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        seq_lens = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

        generation_pbar = tqdm(total=max_new_tokens, desc='Generating sequence', leave=False, disable=disable_pbar)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, cond_vec=cond_vec, start_indices_batch=start_indices_batch, custom_cond_emb=custom_cond_emb)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.full((batch_size, 1), fill_value=PADDING_ID, dtype=torch.long, device=device)
            active_mask = ~finished
            if active_mask.any():
                idx_next[active_mask] = torch.multinomial(probs[active_mask], num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            # Update prev_id and check stopping conditions
            idx_next_squeezed = idx_next.squeeze(-1)
            end_condition = ((prev_id == NEWLINE_ID) & (idx_next_squeezed == NEWLINE_ID)) | (idx_next_squeezed == PADDING_ID)
            # Record sequence lengths when sequences finish
            just_finished = end_condition & (~finished)
            # Exclude the last token that triggered the end condition
            seq_lens[just_finished] = idx.size(1) - 1
            finished = finished | end_condition
            prev_id = idx_next_squeezed
            generation_pbar.update(1)
            if finished.all():
                break
        generation_pbar.close()
        # For sequences that didn't finish, set seq_lens to idx.size(1)
        seq_lens[seq_lens == -1] = idx.size(1)
        # Truncate sequences to their actual length
        max_seq_len = seq_lens.max().item()
        idx_truncated = torch.full((batch_size, max_seq_len), fill_value=Tokenizer().padding_id, dtype=idx.dtype, device=idx.device)
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            idx_truncated[i, :seq_len] = idx[i, :seq_len]
        return idx_truncated

    def expand_wildcard_composition(self, comp_str, ranges=None, default_range=(1, 5)):
        """
        Expands wildcard placeholders in a composition string.
    
        The composition string should contain element symbols (one uppercase letter
        optionally followed by a lowercase letter) immediately followed by a wildcard
        placeholder (one or more lowercase letters). This function replaces each wildcard
        with a random integer chosen from a range specified for that element.
    
        Args:
            comp_str (str): The composition string, e.g. "CexOy" or "AexByCz".
            ranges (dict, optional): Dictionary mapping element symbols (str) to a tuple
                (min, max) representing the range for the random integer. If an element is
                not provided here, the default_range is used.
            default_range (tuple, optional): Default (min, max) range to use if not provided in ranges.
    
        Returns:
            str: The composition string with wildcards replaced by integers.
    
        Example:
            expand_wildcard_composition("CexOy", ranges={'Ce': (1, 5), 'O': (2, 6)})
            might return "Ce3O4".
        """
    
        # Pattern: element symbol (capital letter + optional lowercase) optionally followed by a multiplier (digits) and a valid wildcard ('x', 'y', or 'z').
        pattern = re.compile(r'([A-Z][a-z]?)(\d*)([xyz])')
        wildcard_values = {}  # Stores the base random value for each wildcard letter
        
        def replacer(match):
            element = match.group(1)
            multiplier_str = match.group(2)
            wildcard = match.group(3)
            # Determine multiplier, defaulting to 1 if not provided.
            multiplier = int(multiplier_str) if multiplier_str else 1
            # Sample the random value for this wildcard only once.
            if wildcard not in wildcard_values:
                rmin, rmax = ranges.get(element, default_range) if ranges is not None else default_range
                base_val = random.randint(rmin, rmax)
                wildcard_values[wildcard] = base_val
            else:
                base_val = wildcard_values[wildcard]
            # Multiply the stored base value by the multiplier.
            return f"{element}{multiplier * base_val}"
    
        return pattern.sub(replacer, comp_str)

    @torch.no_grad()
    def generate_custom(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        cond_vec: Optional[torch.Tensor],
        start_indices_batch: List[List[int]],
        temperature: float = 1.0,
        top_k: Optional[float] = None,
        disable_pbar: bool = False,
        custom_cond_emb: Optional[torch.Tensor] = None,
        composition_string: Optional[str] = None,
        composition_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        spacegroup_string: Optional[str] = None,
        cell_a_string: Optional[str] = None,
        cell_b_string: Optional[str] = None,
        cell_c_string: Optional[str] = None,
        cell_alpha_string: Optional[str] = None,
        cell_beta_string: Optional[str] = None,
        cell_gamma_string: Optional[str] = None,
        atoms_string_list: Optional[List[str]] = None,
        exclude_elements: Optional[List[str]] = None,
        exclude_spacegroups: Optional[List[str]] = None,
        include_spacegroups: Optional[List[str]] = None,
    ) -> torch.Tensor:

        """
        -- Single-sequence version --
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        This function takes custom token inputs that it uses to override certain tokens at certain positions.
        """

        # Precompute excluded token ids outside the loop (flattened)
        exclude_ids = []
        if exclude_elements:
            for el in exclude_elements:
                token_ids = ENCODE(TOKENIZE(el))
                if isinstance(token_ids, list):
                    exclude_ids.extend(token_ids)
                else:
                    exclude_ids.append(token_ids)
        if exclude_spacegroups:
            for sp in exclude_spacegroups:
                token_ids = ENCODE(TOKENIZE(sp))
                if isinstance(token_ids, list):
                    exclude_ids.extend(token_ids)
                else:
                    exclude_ids.append(token_ids)
        precomputed_exclude_ids = torch.tensor(exclude_ids, device=self.device) if exclude_ids else None

        prev_id = None
        generation_pbar = tqdm(desc="Generating custom sequence", leave=False, disable=disable_pbar)

        for i in range(max_new_tokens):
            
            # Identify the last token in the current sequence 
            last_token = DECODE([idx[0][-1].item()])
            
            if composition_string is not None and last_token == "data_":
                # If wildcards are present, expand the composition within ranges
                #print("Old comp string:", composition_string)
                composition_string = self.expand_wildcard_composition(composition_string, ranges=composition_ranges)
                #print("New Comp string:", composition_string)
                # Insert composition and newline
                idx_insert = torch.tensor(ENCODE(TOKENIZE(composition_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                idx = torch.cat((idx, idx_insert), dim=1)
            if idx.size(1) > 2:
                second_to_last_token = DECODE([idx[0][-2].item()])

                if (spacegroup_string is not None or include_spacegroups is not None) and second_to_last_token == "_symmetry_space_group_name_H-M" and last_token == " ":
                    if include_spacegroups is not None:
                        # Override the spacegroup_string with random space group from chosen list
                        spacegroup_string = random.choice(include_spacegroups)
                        #print(spacegroup_string)
                    # Insert space group and newline
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(spacegroup_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)
                if cell_a_string is not None and second_to_last_token == "_cell_length_a" and last_token == " ":
                    # Insert cell length a
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(cell_a_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)
                if cell_b_string is not None and last_token == "_cell_length_b":
                    # Insert cell length a
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(cell_b_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)
                if cell_c_string is not None and last_token == "_cell_length_c":
                    # Insert cell length a
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(cell_c_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)
                if cell_alpha_string is not None and last_token == "_cell_angle_alpha":
                    # Insert cell length a
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(cell_alpha_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)
                if cell_beta_string is not None and last_token == "_cell_angle_beta":
                    # Insert cell length a
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(cell_beta_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)
                if cell_gamma_string is not None and last_token == "_cell_angle_gamma":
                    # Insert cell length a
                    idx_insert = torch.tensor(ENCODE(TOKENIZE(cell_gamma_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                    idx = torch.cat((idx, idx_insert), dim=1)

                # Insert atoms
                if atoms_string_list is not None and second_to_last_token == "_atom_site_occupancy" and last_token == "\n":
                    for atom_string in atoms_string_list:
                        idx_insert = torch.tensor(ENCODE(TOKENIZE(atom_string)) + [NEWLINE_ID]).to(device=self.device).unsqueeze(0)
                        idx = torch.cat((idx, idx_insert), dim=1)

            # If the sequence has grown longer than the context length, chop
            idx_context = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size]

            # Forward the model to get logits
            logits, _ = self(
                idx_context,
                cond_vec=cond_vec, 
                start_indices_batch=start_indices_batch,
                custom_cond_emb=custom_cond_emb,
            )

            # Pluck logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Crop the logits using top_k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Apply precomputed exclusions to logits if available
            if precomputed_exclude_ids is not None:
                logits[:, precomputed_exclude_ids] = -float("inf")

            # Apply softmax
            probs = F.softmax(logits, dim=1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop generation if met with two new lines
            if prev_id is not None and prev_id == NEWLINE_ID and idx_next.item() == NEWLINE_ID:
                generation_pbar.close()
                break

            # Stop as soon as the pad-idx is hit
            if idx_next.item() == PADDING_ID:
                idx = idx[:,:-1]
                generation_pbar.close()
                break

            # Update previous id and pbar
            prev_id = idx_next.item()
            generation_pbar.update(1)


        return idx
    
    @torch.no_grad()
    def generate_and_print(self, idx, max_new_tokens, cond_vec=None, start_indices_batch=None, temperature=1.0, top_k=None, custom_cond_emb=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        prev_id = None
            
        for id in idx[0]:
            token = tokenizer.id_to_token[id.item()]
            for char in token:
                sys.stdout.write(char)
                sys.stdout.flush()

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, cond_vec=cond_vec, start_indices_batch=start_indices_batch, custom_cond_emb=custom_cond_emb)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # as soon as <pad> is hit -> end of cif
            if idx_next.item() == PADDING_ID:
                idx = idx[:,:-1]
                break

            # Decode and print
            token_next = tokenizer.id_to_token[idx_next[0].item()]
            for char in token_next:
                sys.stdout.write(char)
                sys.stdout.flush()

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # a sequence of two newlines indicates the end of a CIF file
            if prev_id is not None and prev_id == NEWLINE_ID and idx_next.item() == NEWLINE_ID:
                break
                
            prev_id = idx_next.item()

        return idx
