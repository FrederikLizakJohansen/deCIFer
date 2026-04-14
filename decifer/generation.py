#!/usr/bin/env python3

from typing import List
from warnings import warn

import torch
from torch.nn.utils.rnn import pad_sequence

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    extract_space_group_symbol,
    reinstate_symmetry_loop,
    replace_symmetry_loop_with_P1,
)


TOKENIZER = Tokenizer()
VOCAB_SIZE = TOKENIZER.vocab_size
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]
SPACEGROUP_ID = TOKENIZER.token_to_id["_symmetry_space_group_name_H-M"]
FORMULA_SUM_ID = TOKENIZER.token_to_id["_chemical_formula_sum"]
FORMULA_STRUCTURAL_ID = TOKENIZER.token_to_id["_chemical_formula_structural"]
DECODE = TOKENIZER.decode


def _ensure_long_tensor(sequence) -> torch.Tensor:
    if isinstance(sequence, torch.Tensor):
        return sequence.detach().clone().long()
    return torch.as_tensor(sequence, dtype=torch.long)


def _first_index(sequence: torch.Tensor, token_id: int, start: int = 0) -> int:
    matches = (sequence[start:] == token_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise IndexError(token_id)
    return start + matches[0].item()


def _first_index_any(sequence: torch.Tensor, token_ids, start: int = 0) -> int:
    matches = []
    for token_id in token_ids:
        try:
            matches.append(_first_index(sequence, token_id, start=start))
        except IndexError:
            continue
    if not matches:
        raise IndexError(tuple(token_ids))
    return min(matches)


def _line_end_exclusive(sequence: torch.Tensor, start: int) -> int:
    return _first_index(sequence, NEWLINE_ID, start=start) + 1


def extract_prompt(sequence, device, add_composition: bool = True, add_spacegroup: bool = False):
    sequence = _ensure_long_tensor(sequence)

    try:
        data_line_start = _first_index(sequence, START_ID)
    except IndexError as exc:
        raise ValueError(f"'data_' id: {START_ID} not found in sequence", DECODE(sequence.tolist())) from exc

    prompt_end_exclusive = _line_end_exclusive(sequence, start=data_line_start)
    metadata_search_start = prompt_end_exclusive

    if add_composition:
        try:
            formula_index = _first_index_any(
                sequence,
                [FORMULA_SUM_ID, FORMULA_STRUCTURAL_ID],
                start=metadata_search_start,
            )
            prompt_end_exclusive = max(prompt_end_exclusive, _line_end_exclusive(sequence, start=formula_index))
        except IndexError:
            pass

    if add_spacegroup:
        try:
            spacegroup_index = _first_index(sequence, SPACEGROUP_ID, start=metadata_search_start)
            prompt_end_exclusive = max(prompt_end_exclusive, _line_end_exclusive(sequence, start=spacegroup_index))
        except IndexError:
            pass

    return sequence[:prompt_end_exclusive].to(device=device)


def extract_prompt_batch(sequences, device, add_composition: bool = True, add_spacegroup: bool = False):
    prompts: List[torch.Tensor] = []
    prompt_lengths = []

    for sequence in sequences:
        prompt_ids = extract_prompt(
            sequence,
            device=device,
            add_composition=add_composition,
            add_spacegroup=add_spacegroup,
        )
        prompts.append(prompt_ids)
        prompt_lengths.append(len(prompt_ids))

    padded_prompts = pad_sequence(prompts, batch_first=True, padding_value=PADDING_ID)
    return padded_prompts, prompt_lengths


def strip_padding(token_ids, padding_id: int = PADDING_ID) -> torch.Tensor:
    token_ids = _ensure_long_tensor(token_ids)
    return token_ids[token_ids != padding_id]


def generate_one(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    cond_vec=None,
    start_indices_batch=None,
    temperature: float = 1.0,
    top_k=None,
):
    generated = model.generate(
        prompt.clone(),
        max_new_tokens,
        cond_vec=cond_vec,
        start_indices_batch=start_indices_batch,
        temperature=temperature,
        top_k=top_k,
        disable_pbar=True,
    )
    return strip_padding(generated.squeeze(0).cpu())


def fix_generated_cif(cif_string: str) -> str:
    cif_string = replace_symmetry_loop_with_P1(cif_string)
    spacegroup_symbol = extract_space_group_symbol(cif_string)
    if spacegroup_symbol != "P 1":
        cif_string = reinstate_symmetry_loop(cif_string, spacegroup_symbol)
    return cif_string


def decode_generated_cif(token_ids, tokenizer: Tokenizer = TOKENIZER) -> str:
    return tokenizer.decode(strip_padding(token_ids).tolist())


def decode_and_fix_cif(token_ids, tokenizer: Tokenizer = TOKENIZER) -> str:
    return fix_generated_cif(decode_generated_cif(token_ids, tokenizer=tokenizer))


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("best_model_state", checkpoint.get("best_model"))
    model_args = checkpoint["model_args"]

    renamed_keys = {
        "cond_size": "condition_size",
        "condition_with_mlp_emb": "condition",
    }
    for old_key, new_key in renamed_keys.items():
        if old_key in model_args:
            model_args["use_old_model_format"] = True
            warn(
                f"'{old_key}' is deprecated and has been renamed to '{new_key}'. "
                "Please update your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_args[new_key] = model_args.pop(old_key)

    removed_keys = [
        "use_lora",
        "lora_rank",
        "condition_with_cl_emb",
        "cl_model_ckpt",
        "freeze_condition_embedding",
    ]
    for removed_key in removed_keys:
        if removed_key in model_args:
            warn(
                f"'{removed_key}' is no longer used and will be ignored. "
                "Consider removing it from your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_args.pop(removed_key)

    model_config = DeciferConfig(**model_args)
    model = Decifer(model_config).to(device)
    model.device = device

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    return model
