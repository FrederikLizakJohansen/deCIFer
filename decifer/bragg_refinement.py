"""BraggCalculator refinement configuration and result serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml
from braggcalculator import (
    OptimizationStage,
    RefinementPolicy,
    SpeciesAssignmentConfig,
    refine_structure,
)
from pymatgen.io.cif import CifWriter


@dataclass(frozen=True)
class BraggRefinementConfig:
    enabled: bool
    domain: str
    radiation: str
    wavelength: float
    device: str
    policy_name: str
    policy: RefinementPolicy
    species_assignment: Optional[SpeciesAssignmentConfig] = None

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "domain": self.domain,
            "radiation": self.radiation,
            "wavelength": self.wavelength,
            "device": self.device,
            "policy": {
                "preset": self.policy_name,
                **_jsonable(asdict(self.policy)),
            },
            "species_assignment": (
                None
                if self.species_assignment is None
                else _jsonable(asdict(self.species_assignment))
            ),
        }


def load_bragg_refinement_config(
    path="",
    *,
    enabled=False,
    domain=None,
    radiation=None,
    wavelength=None,
    device=None,
    policy_name=None,
    default_wavelength=1.5406,
    default_device="cpu",
):
    raw = {}
    if path:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if "refinement" in raw:
            raw = raw["refinement"] or {}
        if not isinstance(raw, Mapping):
            raise TypeError("refinement configuration must be a mapping")

    allowed = {
        "enabled",
        "domain",
        "radiation",
        "wavelength",
        "device",
        "policy",
        "species_assignment",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown refinement configuration keys: {unknown}")

    policy_raw = dict(raw.get("policy") or {})
    if policy_name is not None:
        policy_raw["preset"] = policy_name
    selected_policy_name, policy = _policy_from_mapping(policy_raw)
    species_assignment = _species_assignment_from_mapping(
        raw.get("species_assignment")
    )
    selected_domain = domain or raw.get("domain", "q")
    selected_radiation = radiation or raw.get("radiation", "xray")
    selected_wavelength = (
        wavelength
        if wavelength is not None
        else (
            raw.get("wavelength")
            if raw.get("wavelength") is not None
            else default_wavelength
        )
    )
    selected_device = device or raw.get("device") or default_device
    selected_enabled = bool(enabled or raw.get("enabled", bool(path)))

    if selected_domain not in {"q", "two_theta"}:
        raise ValueError("refinement domain must be 'q' or 'two_theta'")
    if selected_radiation not in {"xray", "neutron"}:
        raise ValueError("refinement radiation must be 'xray' or 'neutron'")
    if selected_wavelength is None or float(selected_wavelength) <= 0:
        raise ValueError("refinement wavelength must be positive")

    return BraggRefinementConfig(
        enabled=selected_enabled,
        domain=selected_domain,
        radiation=selected_radiation,
        wavelength=float(selected_wavelength),
        device=str(selected_device),
        policy_name=selected_policy_name,
        policy=policy,
        species_assignment=species_assignment,
    )


def _policy_from_mapping(raw):
    raw = dict(raw or {})
    preset = str(raw.pop("preset", "quick"))
    if preset not in {"quick", "cautious", "robust"}:
        raise ValueError("refinement policy preset must be quick, cautious, or robust")

    stages_raw = raw.pop("stages", None)
    common = {
        key: raw[key]
        for key in (
            "refine_coordinates",
            "occupancy_mode",
            "refine_b_iso",
            "refine_u_aniso",
            "rigid_bodies",
        )
        if key in raw
    }
    if preset == "robust":
        common.update(
            {
                key: raw[key]
                for key in ("likelihood", "restarts")
                if key in raw
            }
        )
    policy = getattr(RefinementPolicy, preset)(**common)
    policy_fields = set(RefinementPolicy.__dataclass_fields__)
    unknown = sorted(set(raw) - policy_fields)
    if unknown:
        raise ValueError(f"unknown RefinementPolicy keys: {unknown}")
    if raw:
        policy = replace(policy, **raw)
    if stages_raw is not None:
        policy = replace(
            policy,
            stages=tuple(_optimization_stage(item) for item in stages_raw),
        )
    return preset, policy


def _optimization_stage(raw):
    if not isinstance(raw, Mapping):
        raise TypeError("each refinement policy stage must be a mapping")
    values = dict(raw)
    values["active"] = tuple(values["active"])
    return OptimizationStage(**values)


def _species_assignment_from_mapping(raw):
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("species_assignment must be a mapping")
    values = dict(raw)
    if not values.pop("enabled", True):
        return None
    unknown = sorted(
        set(values) - set(SpeciesAssignmentConfig.__dataclass_fields__)
    )
    if unknown:
        raise ValueError(f"unknown SpeciesAssignmentConfig keys: {unknown}")
    if "fixed_sites" in values:
        values["fixed_sites"] = tuple(values["fixed_sites"])
    return SpeciesAssignmentConfig(**values)


def refinement_coordinate(q, config):
    q = np.asarray(q, dtype=np.float64)
    if config.domain == "q":
        return q
    argument = q * config.wavelength / (4.0 * np.pi)
    if np.any(argument > 1.0 + 1e-12):
        raise ValueError("Q coordinate exceeds the refinement wavelength limit")
    return np.degrees(2.0 * np.arcsin(np.clip(argument, 0.0, 1.0)))


def run_bragg_refinement(
    q,
    observed,
    structure,
    config,
    *,
    initial_fit_statistics=None,
    refine_fn=refine_structure,
):
    """Run one isolated refinement and return its result and JSON record."""
    record = {
        "schema": "decifer.bragg-refinement/v1",
        "attempted": True,
        "succeeded": False,
        "configuration": config.to_dict(),
        "initial_fit_statistics": _jsonable(initial_fit_statistics or {}),
    }
    try:
        coordinate = refinement_coordinate(q, config)
        pattern = np.column_stack(
            [coordinate, np.asarray(observed, dtype=np.float64)]
        )
        result = refine_fn(
            pattern=pattern,
            structure=structure,
            wavelength=config.wavelength,
            radiation=config.radiation,
            domain=config.domain,
            policy=config.policy,
            species_assignment=config.species_assignment,
            device=config.device,
        )
        record.update(_serialize_result(result))
        record["succeeded"] = True
        return result, record
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return None, record


def failed_refinement_record(config, message, *, initial_fit_statistics=None):
    return {
        "schema": "decifer.bragg-refinement/v1",
        "attempted": False,
        "succeeded": False,
        "configuration": config.to_dict(),
        "initial_fit_statistics": _jsonable(initial_fit_statistics or {}),
        "error": {
            "type": "InvalidCandidate",
            "message": str(message),
        },
    }


def _serialize_result(result):
    return {
        "status": result.status,
        "convergence": _jsonable(result.convergence),
        "refined_fit_statistics": _jsonable(result.fit_statistics),
        "refined_parameters": [_jsonable(parameter) for parameter in result.parameters],
        "warnings": list(result.warnings),
        "objective_history": _jsonable(result.objective_history),
        "stage_history": list(result.stage_history),
        "coordinate": _jsonable(result.coordinate),
        "observed": _jsonable(result.observed),
        "calculated_profile": _jsonable(result.calculated),
        "residual": _jsonable(result.residual),
        "starting_cif": str(CifWriter(result.starting_structure, symprec=None)),
        "refined_cif": result.refined_cif,
        "diagnostics": _jsonable(result.diagnostics),
        "provenance": _jsonable(result.provenance),
        "species_assignment": _serialize_species_assignment(
            result.species_assignments
        ),
    }


def _serialize_species_assignment(result):
    if result is None:
        return None
    return {
        "search_mode": result.search_mode,
        "generated_count": result.generated_count,
        "evaluated_count": result.evaluated_count,
        "deduplicated_count": result.deduplicated_count,
        "truncated": result.truncated,
        "ambiguity_tolerance": result.ambiguity_tolerance,
        "indistinguishable_assignments": list(
            result.indistinguishable_assignments
        ),
        "warnings": list(result.warnings),
        "target_composition": _jsonable(result.target_composition),
        "sites": [_jsonable(site) for site in result.sites],
        "candidates": [
            {
                "assignment_id": candidate.assignment_id,
                "sites": [_jsonable(site) for site in candidate.sites],
                "screening_score": candidate.screening_score,
                "continuous_score": candidate.continuous_score,
                "convergence": _jsonable(candidate.convergence),
                "indistinguishable": candidate.indistinguishable,
            }
            for candidate in result.candidates
        ],
    }


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)
