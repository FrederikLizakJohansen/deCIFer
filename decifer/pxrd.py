#!/usr/bin/env python3

from typing import Optional, Tuple

import torch


def nyquist_qstep(fwhm: float, points_per_fwhm: float = 2.0) -> float:
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    if points_per_fwhm <= 0:
        raise ValueError("points_per_fwhm must be positive")
    return float(fwhm) / float(points_per_fwhm)


def discrete_to_continuous_xrd(
    batch_q,
    batch_iq,
    qmin: float = 0.0,
    qmax: float = 10.0,
    qstep: Optional[float] = 0.01,
    nyquist_points_per_fwhm: Optional[float] = None,
    fwhm_range: Tuple[float, float] = (0.01, 0.5),
    eta_range: Tuple[float, float] = (0.5, 0.5),
    noise_range: Optional[Tuple[float, float]] = (0.001, 0.05),
    intensity_scale_range: Optional[Tuple[float, float]] = (0.95, 1.0),
    mask_prob: Optional[float] = 0.1,
    q_shift_range: Optional[Tuple[float, float]] = None,
    q_scale_range: Optional[Tuple[float, float]] = None,
    peak_intensity_jitter_range: Optional[Tuple[float, float]] = None,
    peak_dropout_prob: Optional[float] = None,
    background_range: Optional[Tuple[float, float]] = None,
    impurity_peak_count_range: Optional[Tuple[int, int]] = None,
    impurity_intensity_range: Optional[Tuple[float, float]] = None,
    particle_size_range: Optional[Tuple[float, float]] = None,
    peak_asymmetry_range: Optional[Tuple[float, float]] = None,
    final_normalize: bool = False,
    **kwargs,
):
    if nyquist_points_per_fwhm is not None:
        qstep = nyquist_qstep(fwhm_range[0], nyquist_points_per_fwhm)
    if qstep is None or qstep <= 0:
        raise ValueError("qstep must be positive unless nyquist_points_per_fwhm is set")

    q_cont = torch.arange(qmin, qmax, qstep, dtype=batch_q.dtype, device=batch_q.device)
    batch_size = batch_q.shape[0]
    num_q_points = q_cont.shape[0]

    batch_q = batch_q.clone()
    batch_iq = batch_iq.clone()
    valid_peaks = batch_q != 0

    if q_scale_range is not None:
        q_scale = _uniform(batch_size, 1, range_=q_scale_range, like=batch_q)
        batch_q = batch_q * q_scale

    if q_shift_range is not None:
        q_shift = _uniform(batch_size, 1, range_=q_shift_range, like=batch_q)
        batch_q = batch_q + q_shift

    if intensity_scale_range is not None:
        intensity_scale = _uniform(batch_size, 1, range_=intensity_scale_range, like=batch_iq)
        batch_iq = batch_iq * intensity_scale

    if peak_intensity_jitter_range is not None:
        jitter = _uniform(*batch_iq.shape, range_=peak_intensity_jitter_range, like=batch_iq)
        batch_iq = batch_iq * jitter

    if peak_dropout_prob is not None and peak_dropout_prob > 0:
        keep = torch.rand(batch_iq.shape, dtype=batch_iq.dtype, device=batch_iq.device) > peak_dropout_prob
        batch_iq = batch_iq * keep

    fwhm = _uniform(batch_size, 1, 1, range_=fwhm_range, like=batch_q)
    if particle_size_range is not None:
        particle_size = _uniform(batch_size, 1, 1, range_=particle_size_range, like=batch_q)
        particle_fwhm = torch.where(particle_size > 0, 0.9 / particle_size, torch.zeros_like(particle_size))
        fwhm = fwhm + particle_fwhm

    eta = _uniform(batch_size, 1, 1, range_=eta_range, like=batch_q)
    q_cont_expanded = q_cont.view(1, num_q_points, 1)
    batch_q_expanded = batch_q.unsqueeze(1)
    delta_q = q_cont_expanded - batch_q_expanded

    fwhm_eff = fwhm
    if peak_asymmetry_range is not None:
        asymmetry = _uniform(batch_size, 1, 1, range_=peak_asymmetry_range, like=batch_q)
        fwhm_eff = torch.where(delta_q >= 0, fwhm * (1 + torch.clamp(asymmetry, min=0)), fwhm * (1 + torch.clamp(-asymmetry, min=0)))

    iq_cont = _pseudo_voigt_sum(delta_q, batch_iq.unsqueeze(1), valid_peaks.unsqueeze(1), fwhm_eff, eta)
    iq_cont /= (iq_cont.max(dim=1, keepdim=True)[0] + 1e-16)

    if impurity_peak_count_range is not None and impurity_peak_count_range[1] > 0:
        iq_cont = iq_cont + _impurity_signal(
            q_cont,
            batch_size,
            impurity_peak_count_range,
            impurity_intensity_range or (0.01, 0.1),
            fwhm,
            eta,
        )

    if background_range is not None:
        background_scale = _uniform(batch_size, 1, range_=background_range, like=iq_cont)
        q_norm = (q_cont - q_cont.min()) / (q_cont.max() - q_cont.min() + 1e-16)
        slope = _uniform(batch_size, 1, range_=(-1.0, 1.0), like=iq_cont)
        background = background_scale * torch.clamp(1.0 + slope * (q_norm.view(1, -1) - 0.5), min=0.0)
        iq_cont = iq_cont + background

    if noise_range is not None:
        noise_scale = _uniform(batch_size, 1, range_=noise_range, like=iq_cont)
        iq_cont = iq_cont + torch.randn(batch_size, num_q_points, dtype=iq_cont.dtype, device=iq_cont.device) * noise_scale

    if mask_prob is not None and mask_prob > 0:
        mask = (torch.rand(batch_size, num_q_points, dtype=iq_cont.dtype, device=iq_cont.device) > mask_prob).float()
        iq_cont = iq_cont * mask

    iq_cont = torch.clamp(iq_cont, min=0.0)
    if final_normalize:
        iq_cont /= (iq_cont.max(dim=1, keepdim=True)[0] + 1e-16)

    return {"q": q_cont, "iq": iq_cont}


def _uniform(*shape, range_, like):
    return torch.empty(*shape, dtype=like.dtype, device=like.device).uniform_(*range_)


def _pseudo_voigt_sum(delta_q, peak_iq, valid_peaks, fwhm, eta):
    sigma_gauss = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, dtype=delta_q.dtype, device=delta_q.device))))
    gamma_lorentz = fwhm / 2
    gaussian_component = torch.exp(-0.5 * (delta_q / sigma_gauss) ** 2)
    lorentzian_component = 1 / (1 + (delta_q / gamma_lorentz) ** 2)
    pseudo_voigt = eta * lorentzian_component + (1 - eta) * gaussian_component
    return (pseudo_voigt * peak_iq * valid_peaks.float()).sum(dim=2)


def _impurity_signal(q_cont, batch_size, count_range, intensity_range, fwhm, eta):
    max_count = int(count_range[1])
    min_count = int(count_range[0])
    counts = torch.randint(min_count, max_count + 1, (batch_size,), device=q_cont.device)
    impurity_q = torch.empty(batch_size, max_count, dtype=q_cont.dtype, device=q_cont.device).uniform_(float(q_cont.min()), float(q_cont.max()))
    impurity_iq = torch.empty(batch_size, max_count, dtype=q_cont.dtype, device=q_cont.device).uniform_(*intensity_range)
    count_mask = torch.arange(max_count, device=q_cont.device).view(1, -1) < counts.view(-1, 1)
    impurity_iq = impurity_iq * count_mask.float()
    delta_q = q_cont.view(1, -1, 1) - impurity_q.unsqueeze(1)
    return _pseudo_voigt_sum(delta_q, impurity_iq.unsqueeze(1), count_mask.unsqueeze(1), fwhm, eta)
