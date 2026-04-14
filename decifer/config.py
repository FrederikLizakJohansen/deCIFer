#!/usr/bin/env python3

import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml


T = TypeVar("T")


def _construct_dataclass_defaults(config_cls: Type[T]) -> T:
    kwargs = {}
    for field_info in fields(config_cls):
        if field_info.default is not MISSING:
            kwargs[field_info.name] = field_info.default
        elif field_info.default_factory is not MISSING:
            kwargs[field_info.name] = field_info.default_factory()
    return config_cls(**kwargs)


def _filter_known_keys(config_cls: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    valid_keys = {field_info.name for field_info in fields(config_cls)}
    unknown_keys = set(data) - valid_keys
    if unknown_keys:
        unknown_keys_str = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown config keys for {config_cls.__name__}: {unknown_keys_str}")
    return {key: value for key, value in data.items() if key in valid_keys}


def load_yaml_dict(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    return data


def dataclass_to_dict(config_obj: T) -> Dict[str, Any]:
    return asdict(config_obj)


def load_dataclass_config(
    config_cls: Type[T],
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> T:
    config_data = dataclass_to_dict(_construct_dataclass_defaults(config_cls))

    if config_path:
        yaml_data = _filter_known_keys(config_cls, load_yaml_dict(config_path))
        config_data.update(yaml_data)

    if overrides:
        override_data = _filter_known_keys(
            config_cls,
            {key: value for key, value in overrides.items() if value is not None},
        )
        config_data.update(override_data)

    return config_cls(**config_data)


@dataclass
class EvaluateConfig:
    model_ckpt: Optional[str] = None
    root: str = "./"
    num_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 1) - 1))
    dataset_path: Optional[str] = None
    dataset_split: str = "test"
    out_folder: Optional[str] = None
    debug_max: Optional[int] = None
    debug: bool = False
    add_composition: bool = False
    add_spacegroup: bool = False
    max_new_tokens: int = 1000
    dataset_name: str = "default_dataset"
    model_name: str = "default_model"
    num_reps: int = 1
    override: bool = False
    condition: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    add_noise: Optional[float] = None
    add_broadening: Optional[float] = None
    default_fwhm: float = 0.05
    clean_fwhm: float = 0.05
    qmin: float = 0.0
    qmax: float = 10.0
    qstep: float = 0.01
    wavelength: str = "CuKa"
    eta: float = 0.5


@dataclass
class AblationConfig:
    dataset_path: Optional[str] = None
    dataset_split: str = "test"
    model_path: Optional[str] = None
    params_dict: Dict[str, Any] = field(default_factory=dict)
    default_params_dict: Optional[Dict[str, Any]] = None
    batch_size: int = 1
    n_repeats: int = 1
    max_new_tokens: int = 3076
    add_composition: bool = False
    add_spacegroup: bool = False
    crystal_system: Optional[str] = None
    target_elements: Optional[list] = None
    element_match_mode: str = "exact"
    element_count: Optional[int] = None
    seed: int = 100
    use_multi_phase: bool = False
    combinatory: bool = False
    output: str = "experiment_results.pkl"
    temperature: float = 1.0
    top_k: Optional[int] = None


@dataclass
class TrainWorkflowConfig:
    out_dir: str = "out"
    eval_interval: int = 250
    log_interval: int = 1
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = False
    init_from: str = "scratch"
    dataset: str = ""
    gradient_accumulation_steps: int = 40
    batch_size: int = 64
    cond_size: int = 1000
    accumulative_pbar: bool = False
    num_workers_dataloader: int = 0
    block_size: int = 1024
    vocab_size: int = 372
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    boundary_masking: bool = True
    condition: bool = False
    condition_embedder_hidden_layers: List[int] = field(default_factory=lambda: [512])
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
    learning_rate: float = 6e-4
    max_iters: int = 50_000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    device: str = "cuda"
    dtype: str = "float16"
    compile: bool = False
    validate: bool = False
    seed: int = 1337
    early_stopping_patience: int = 50


@dataclass
class RunProtocolConfig:
    model_path: Optional[str] = None
    zip_path: Optional[str] = None
    debug_max: Optional[int] = None
    n_trials: int = 25
    suffix: str = "default"
    target_files: List[str] = field(
        default_factory=lambda: [
            "scan-4907_mean.xy",
            "scan-4911_mean.xy",
            "scan-4912_mean.xy",
            "scan-4919_mean.xy",
        ]
    )
    background_file: str = "scan-4903_mean.xy"
    wavelength: Optional[str] = None
    q_min_crop: float = 1.5
    q_max_crop: float = 8.0
    protocols: List[List[Any]] = field(
        default_factory=lambda: [
            [{}, "none"],
            [{"spacegroup": "Fm-3m_sg"}, "Fm-3m"],
            [{"crystal_systems": [7]}, "Cubic"],
            [{"composition": "Ce1O2"}, "Ce1O2"],
            [{"composition": "Ce2O4"}, "Ce2O4"],
            [{"composition": "Ce4O8"}, "Ce4O8"],
            [{"composition": "Ce4O8", "spacegroup": "Fm-3m_sg"}, "Ce4O8_Fm-3m"],
        ]
    )
