#!/usr/bin/env python3

import re
import math
from typing import List, Optional, Tuple
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser, CifBlock
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure

import warnings
warnings.filterwarnings('ignore')

from periodictable import elements as pelements

def element_to_atomic_number(element):
    try:
        return pelements.symbol(element).number
    except KeyError:
        print(f"Invalid element symbol: {element}")
        return None

def space_group_to_crystal_system(space_group):
    try:
        if 1 <= space_group <= 2:
            return 1  # Triclinic
        elif 3 <= space_group <= 15:
            return 2  # Monoclinic
        elif 16 <= space_group <= 74:
            return 3  # Orthorhombic
        elif 75 <= space_group <= 142:
            return 4  # Tetragonal
        elif 143 <= space_group <= 167:
            return 5  # Trigonal
        elif 168 <= space_group <= 194:
            return 6  # Hexagonal
        elif 195 <= space_group <= 230:
            return 7  # Cubic
        else:
            print(f"Invalid space group: {space_group}")
            return 0
    except:
        return 0

def print_hdf5_structure(file_path):
    """
    This function opens an HDF5 file, prints its structure (groups and datasets),
    and allows access to the data.

    Parameters:
    file_path (str): Path to the HDF5 file.
    """
    try:
        with h5py.File(file_path, 'r') as hdf_file:
            # Recursive function to print group structure
            def print_group(name, node):
                if isinstance(node, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {node.shape}, Dtype: {node.dtype}")
                elif isinstance(node, h5py.Group):
                    print(f"Group: {name}")
                    
            # Visit every node in the file and print details
            hdf_file.visititems(print_group)

            # Return the HDF5 file object to explore it further outside the function
            return hdf_file
    except Exception as e:
        print(f"Error: {e}")

def discrete_to_continuous_xrd(
    batch_q,
    batch_iq,
    qmin: float = 0.0,
    qmax: float = 10.0,
    qstep: float = 0.01,
    fwhm_range: Tuple[float, float] = (0.01, 0.5),
    eta_range: Tuple[float, float] = (0.5, 0.5),  # Mixing range for pseudo-Voigt, 1.0 is fully Lorentzian
    noise_range: Optional[Tuple[float, float]] = None,
    intensity_scale_range: Optional[Tuple[float, float]] = None,
    mask_prob: Optional[float] = None,
    **kwargs, # Ignoring
):
    """
    Converts discrete XRD data (q, iq) to a continuous representation using pseudo-Voigt peak broadening.

    Args:
        batch_q (Tensor): A tensor containing discrete q-values for each sample, 
            with shape (batch_size, max_num_peaks).
        batch_iq (Tensor): A tensor containing discrete iq-values (intensities) 
            corresponding to batch_q, with the same shape.
        qmin (float, optional): The minimum q-value for the continuous grid. Default is 0.0.
        qmax (float, optional): The maximum q-value for the continuous grid. Default is 10.0.
        qstep (float, optional): The step size for the continuous q grid. Default is 0.01.
        fwhm_range (Tuple[float, float], optional): The range of full-width at half-maximum 
            (FWHM) values for peak broadening. Default is (0.01, 0.5).
        eta_range (Tuple[float, float], optional): The range of mixing values for pseudo-Voigt broadening. 
            1.0 is fully Lorentzian. Default is (0.5, 0.5).
        noise_range (Tuple[float, float], optional): The range for random noise added to the 
            continuous intensities. Default is (0.001, 0.05).
        intensity_scale_range (Tuple[float, float], optional): The range for random scaling of 
            iq values. Default is (0.95, 1.0).
        mask_prob (float, optional): The probability of masking random regions of the 
            continuous intensities. Default is 0.1.

    Returns:
        dict: A dictionary with the following keys:
            - 'q' (Tensor): The continuous q-values with shape (num_q_points,).
            - 'iq' (Tensor): The continuous iq-values with shape (batch_size, num_q_points).

    Notes:
        - The function applies pseudo-Voigt broadening to the input peaks and adds random noise.
        - Peak intensities are normalized, and negative values are clipped to zero.
    """
    # Generate q_cont based on qmin, qmax, and qstep
    q_cont = torch.arange(qmin, qmax, qstep)  # Shape: (num_q_points,)
    batch_size = batch_q.shape[0]
    num_q_points = q_cont.shape[0]

    # Sample random FWHM, eta, noise, and intensity scale values for each sample
    fwhm = torch.empty(batch_size, 1, 1).uniform_(*fwhm_range)  # Shape: (batch_size, 1, 1)
    eta = torch.empty(batch_size, 1, 1).uniform_(*eta_range)    # Shape: (batch_size, 1, 1)

    # Apply intensity scaling to peak intensities
    if intensity_scale_range is not None:
        intensity_scale = torch.empty(batch_size, 1).uniform_(*intensity_scale_range)  # Shape: (batch_size, 1)
        batch_iq = batch_iq * intensity_scale

    # Convert FWHM to standard deviations
    sigma_gauss = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))
    gamma_lorentz = fwhm / 2

    # Expand dimensions for broadcasting
    q_cont_expanded = q_cont.view(1, num_q_points, 1)            # Shape: (1, num_q_points, 1)
    batch_q_expanded = batch_q.unsqueeze(1)                      # Shape: (batch_size, 1, max_num_peaks)
    delta_q = q_cont_expanded - batch_q_expanded                 # Shape: (batch_size, num_q_points, max_num_peaks)

    # Compute Gaussian and Lorentzian components
    gaussian_component = torch.exp(-0.5 * (delta_q / sigma_gauss) ** 2)
    lorentzian_component = 1 / (1 + (delta_q / gamma_lorentz) ** 2)
    pseudo_voigt = eta * lorentzian_component + (1 - eta) * gaussian_component

    # Apply peak intensities and filter out padded values
    batch_iq_expanded = batch_iq.unsqueeze(1)                    # Shape: (batch_size, 1, max_num_peaks)
    valid_peaks = (batch_q_expanded != 0).float()                # Shape: (batch_size, 1, max_num_peaks)
    iq_cont = (pseudo_voigt * batch_iq_expanded * valid_peaks).sum(dim=2)  # Shape: (batch_size, num_q_points)

    # Normalize
    iq_cont /= (iq_cont.max(dim=1, keepdim=True)[0] + 1e-16)

    # Add random noise
    if noise_range is not None:
        noise_scale = torch.empty(batch_size, 1).uniform_(*noise_range)  # Shape: (batch_size, 1)
        noise = torch.randn(batch_size, num_q_points) * noise_scale
        iq_cont += noise

    # Apply random masking
    if mask_prob is not None:
        mask = (torch.rand(batch_size, num_q_points) > mask_prob).float()
        iq_cont *= mask
    
    # Clip to ensure non-negative intensities
    iq_cont = torch.clamp(iq_cont, min=0.0)

    return {'q': q_cont, 'iq': iq_cont}

def generate_continuous_xrd_from_cif(
    cif_string,
    structure_name: str = 'null',
    wavelength: str = 'CuKa',
    qmin: float = 0.0,
    qmax: float = 10.0,
    qstep: float = 0.01,
    fwhm_range: Tuple[float, float] = (0.01, 0.5),
    eta_range: Tuple[float, float] = (0.5, 0.5),  # Mixing range for pseudo-Voigt, 1.0 is fully Lorentzian
    noise_range: Optional[Tuple[float, float]] = None,
    intensity_scale_range: Optional[Tuple[float, float]] = None,
    mask_prob: Optional[float] = None,
    debug: bool = False
):
    """
    Generates a continuous XRD pattern from a CIF structure using pseudo-Voigt peak broadening.

    Args:
        cif_string (str): The CIF file content as a string.
        structure_name (str, optional): An optional name for the structure, used for debugging. Default is 'null'.
        wavelength (str, optional): The wavelength of the X-ray source. Default is 'CuKa'.
        qmin (float, optional): The minimum q-value for the continuous grid. Default is 0.0.
        qmax (float, optional): The maximum q-value for the continuous grid. Default is 10.0.
        qstep (float, optional): The step size for the continuous q grid. Default is 0.01.
        fwhm_range (Tuple[float, float], optional): The range of full-width at half-maximum 
            (FWHM) values for peak broadening. Default is (0.01, 0.5).
        eta_range (Tuple[float, float], optional): The range of mixing values for pseudo-Voigt broadening. 
            1.0 is fully Lorentzian. Default is (0.5, 0.5).
        noise_range (Tuple[float, float], optional): The range for random noise added to the 
            continuous intensities. Default is (0.001, 0.05).
        intensity_scale_range (Tuple[float, float], optional): The range for random scaling of 
            iq values. Default is (0.95, 1.0).
        mask_prob (float, optional): The probability of masking random regions of the 
            continuous intensities. Default is 0.1.
        debug (bool, optional): If True, prints debugging information for errors. Default is False.

    Returns:
        dict or None: A dictionary with the following keys, or None if an error occurs:
            - 'q' (ndarray): The continuous q-values with shape (num_q_points,).
            - 'iq' (ndarray): The continuous iq-values with shape (num_q_points,).
            - 'q_disc' (ndarray): The discrete q-values derived from the CIF structure.
            - 'iq_disc' (ndarray): The discrete iq-values derived from the CIF structure.

    Notes:
        - The function parses the CIF string to extract the structure and calculates the discrete XRD pattern.
        - It converts the discrete pattern to a continuous representation using pseudo-Voigt broadening.
        - Continuous intensities are normalized, and negative values are clipped to zero.
        - If an error occurs (e.g., invalid CIF string), the function returns None and optionally logs the error.
    """

    try:
        # Parse the CIF string to get the structure
        structure = Structure.from_str(cif_string, fmt="cif")
        
        # Initialize the XRD calculator using the specified wavelength
        xrd_calculator = XRDCalculator(wavelength=wavelength)
        
        # Calculate the XRD pattern from the structure
        max_q = ((4 * np.pi) / xrd_calculator.wavelength) * np.sin(np.radians(180/2))
        if qmax >= max_q:
            two_theta_range = None
        else:
            tth_min = np.degrees(2 * np.arcsin((qmin * xrd_calculator.wavelength) / (4 * np.pi)))
            tth_max = np.degrees(2 * np.arcsin((qmax * xrd_calculator.wavelength) / (4 * np.pi)))
            two_theta_range = (tth_min, tth_max)
        xrd_pattern = xrd_calculator.get_pattern(structure, two_theta_range=two_theta_range)
    
    except Exception as e:
        if debug:
            print(f"Error processing {structure_name}: {e}")
        return None

    # Convert 2θ (xrd_pattern.x) to Q (momentum transfer)
    theta_radians = torch.tensor(np.radians(xrd_pattern.x / 2), dtype=torch.float32)
    q_disc = 4 * np.pi * torch.sin(theta_radians) / xrd_calculator.wavelength
    iq_disc = torch.tensor(xrd_pattern.y, dtype=torch.float32)
    
    # Apply intensity scaling to discrete peak intensities
    if intensity_scale_range is not None:
        intensity_scale = torch.empty(1).uniform_(*intensity_scale_range).item()
        iq_disc *= intensity_scale

    # Define the continuous Q grid
    q_cont = torch.arange(qmin, qmax, qstep, dtype=torch.float32)
    iq_cont = torch.zeros_like(q_cont)

    # Sample a random FWHM, eta, noise scale, and intensity scale
    fwhm = torch.empty(1).uniform_(*fwhm_range).item()
    eta = torch.empty(1).uniform_(*eta_range).item()

    # Convert FWHM to standard deviations for Gaussian and Lorentzian parts
    sigma_gauss = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))
    gamma_lorentz = fwhm / 2

    # Vectorized pseudo-Voigt broadening over all peaks
    delta_q = q_cont.unsqueeze(1) - q_disc.unsqueeze(0)  # Shape: (len(q_cont), len(q_disc))
    
    # Gaussian and Lorentzian components
    gaussian_component = torch.exp(-0.5 * (delta_q / sigma_gauss) ** 2)
    lorentzian_component = 1 / (1 + (delta_q / gamma_lorentz) ** 2)
    pseudo_voigt = eta * lorentzian_component + (1 - eta) * gaussian_component

    # Apply peak intensities and sum over peaks
    iq_cont = (pseudo_voigt * iq_disc).sum(dim=1)

    # Normalize the continuous intensities
    iq_cont /= (iq_cont.max() + 1e-16)
    
    # Add random noise
    if noise_range is not None:
        noise_scale = torch.empty(1).uniform_(*noise_range).item()
        iq_cont += torch.randn_like(iq_cont) * noise_scale

    # Apply random masking
    if mask_prob is not None:
        mask = (torch.rand_like(iq_cont) > mask_prob).float()
        iq_cont *= mask

    # Clip to ensure non-negative intensities
    iq_cont = torch.clamp(iq_cont, min=0.0)

    return {'q': q_cont.numpy(), 'iq': iq_cont.numpy(), 'q_disc': q_disc.numpy(), 'iq_disc': iq_disc.numpy()}

def pxrd_from_cif(
    cif_string,
    structure_name: str = 'null',
    wavelength: str = 'CuKa',
    qmin: float = 0.0,
    qmax: float = 10.0,
    qstep: float = 0.01,
    base_fwhm: float = 0.05,
    eta: float = 0.5,
    noise: Optional[float] = None,
    random_mask_prob: Optional[float] = None,
    particle_size: Optional[float] = None,
    peak_asymmetry: Optional[float] = None,
    peak_redaction_prob: Optional[float] = None,
    chebychev_order: int = 0,
    chebychev_norm_coeffs: Optional[List[float]] = None,
    preferred_orientation_range: Optional[Tuple[float, float]] = None,
    phase_scales: Optional[List[float]] = None,
    q_shift: float = 0.0,
    q_scale: float = 1.0,
    mask_ranges: Optional[List[Tuple[float, float]]] = None,
    debug: bool = False
):
    try:
        # Allow for multiple phases: if cif_string is not a list, wrap it in a list.
        if not isinstance(cif_string, list):
            cif_list = [cif_string]
        else:
            cif_list = cif_string

        # Create a continuous Q grid
        q_cont = torch.arange(qmin, qmax, qstep, dtype=torch.float32)
        overall_iq_cont = torch.zeros_like(q_cont)

        # Lists to store discrete peak data for each phase
        q_disc_list = []
        iq_disc_list = []

        # Loop over each phase
        for phase_index, cif in enumerate(cif_list):
            try:
                structure = Structure.from_str(cif, fmt="cif")
                xrd_calculator = XRDCalculator(wavelength=wavelength)
                
                # Determine two_theta_range based on qmin and qmax
                max_q = ((4 * np.pi) / xrd_calculator.wavelength) * np.sin(np.radians(90))
                if qmax >= max_q:
                    two_theta_range = None
                else:
                    tth_min = np.degrees(2 * np.arcsin((qmin * xrd_calculator.wavelength) / (4 * np.pi)))
                    tth_max = np.degrees(2 * np.arcsin((qmax * xrd_calculator.wavelength) / (4 * np.pi)))
                    two_theta_range = (tth_min, tth_max)
                
                pattern = xrd_calculator.get_pattern(structure, two_theta_range=two_theta_range)
            except Exception as e:
                if debug:
                    print(f"Error processing {structure_name} (phase {phase_index}): {e}")
                return None

            # Convert 2θ to Q for discrete peaks
            theta = np.radians(np.array(pattern.x) / 2)
            # Ensure wavelength is a float (e.g., CuKa -> 1.5418 Å) if needed; here we assume xrd_calculator.wavelength is numeric.
            wavelength_val = float(xrd_calculator.wavelength)
            q_disc = torch.tensor(4 * np.pi * np.sin(theta) / wavelength_val, dtype=torch.float32)
            iq_disc = torch.tensor(pattern.y, dtype=torch.float32)
            # Apply preferred orientation: scale each peak by a random factor from the given range.
            if preferred_orientation_range is not None:
                po_factors = torch.empty_like(iq_disc).uniform_(*preferred_orientation_range)
                iq_disc *= po_factors

            # Apply redaction of peaks (suppress some peaks)
            if peak_redaction_prob is not None:
                redaction_mask = (torch.rand_like(iq_disc) > peak_redaction_prob).float()
                iq_disc *= redaction_mask

            q_disc = (q_disc + q_shift) * q_scale

            # Save discrete peaks for output
            q_disc_list.append(q_disc)
            iq_disc_list.append(iq_disc)

            # Compute Gaussian and Lorentzian widths (from FWHM)
            sigma_gauss = base_fwhm / (2 * np.sqrt(2 * np.log(2)))
            gamma_lorentz = base_fwhm / 2  # base Lorentzian width

            # --- Particle Size Broadening ---
            # If particle_size is provided, add additional Lorentzian broadening per peak.
            if particle_size is not None:
                # Compute theta for each discrete peak: sin(theta) = q * wavelength / (4π)
                # Clamp the argument to [-1, 1] to avoid errors.
                sin_theta = (q_disc * wavelength_val / (4 * np.pi)).clamp(max=1.0)
                theta_i = torch.as_tensor(np.arcsin(sin_theta.numpy()))
                # Additional broadening from particle size (using Scherrer constant K ~ 0.9)
                gamma_size = 0.9 * wavelength_val / (particle_size * torch.cos(theta_i))
                # Update Lorentzian width per peak (will be a vector matching q_disc)
                gamma_lorentz = gamma_lorentz + gamma_size
                # Optionally, also apply to the Gaussian component:
                sigma_size = gamma_size / (2 * np.sqrt(2 * np.log(2)))
                sigma_gauss = torch.sqrt(sigma_gauss**2 + sigma_size**2)

            # --- Peak Asymmetry ---
            asymmetry = 0.0 if peak_asymmetry is None else peak_asymmetry

            # --- Convolution to Continuous Pattern ---
            # Create a (n_q x n_peaks) difference grid
            delta_q = q_cont.unsqueeze(1) - q_disc.unsqueeze(0)

            # For asymmetry, choose different effective widths on each side of each peak.
            sigma_eff = torch.where(delta_q < 0, sigma_gauss * (1 + asymmetry), sigma_gauss * (1 - asymmetry))
            # gamma_lorentz might be a scalar or vector; if vector, unsqueeze to match dimensions.
            if torch.numel(torch.tensor(gamma_lorentz)) > 1:
                gamma_base = gamma_lorentz.unsqueeze(0)  # shape (1, n_peaks)
            else:
                gamma_base = torch.tensor(gamma_lorentz).unsqueeze(0)
            gamma_eff = torch.where(delta_q < 0, gamma_base * (1 + asymmetry), gamma_base * (1 - asymmetry))

            # Compute the Gaussian and Lorentzian components
            gaussian_component = torch.exp(-0.5 * (delta_q / sigma_eff) ** 2)
            lorentzian_component = 1 / (1 + (delta_q / gamma_eff) ** 2)
            pseudo_voigt = eta * lorentzian_component + (1 - eta) * gaussian_component

            # Sum the contributions of each (redacted, oriented) discrete peak
            phase_iq_cont = (pseudo_voigt * iq_disc).sum(dim=1)

            # Normalize the phase contribution
            phase_iq_cont /= (phase_iq_cont.max() + 1e-16)
            
            # --- Chebychev Background ---
            if chebychev_order > 0 and chebychev_norm_coeffs is not None:
                # Scale q to [-1, 1]
                x_scaled = (2 * q_cont - (qmin + qmax)) / (qmax - qmin)
                background = torch.zeros_like(q_cont)
                chebychev_norm_coeffs = [1] + chebychev_norm_coeffs
                for n in range(chebychev_order + 1):
                    # T_n(x) = cos(n * arccos(x))
                    # Clamp x to [-1, 1] to avoid NaNs.
                    background += chebychev_norm_coeffs[n] * torch.cos(n * torch.acos(x_scaled.clamp(-1, 1)))
                phase_iq_cont += background
            
            # Add random noise if specified
            if noise is not None:
                phase_iq_cont += torch.randn_like(phase_iq_cont) * noise

            # Weight the phase by its scale (if provided)
            phase_weight = 1.0
            if phase_scales is not None and phase_index < len(phase_scales):
                phase_weight = phase_scales[phase_index]
            overall_iq_cont += phase_weight * phase_iq_cont

        # Optionally, apply a random mask to the continuous pattern
        if random_mask_prob is not None:
            mask = (torch.rand_like(overall_iq_cont) > random_mask_prob).float()
            overall_iq_cont *= mask

        # Ensure non-negative intensities
        overall_iq_cont = torch.clamp(overall_iq_cont, min=0.0)

        # --- Masking Specific q Ranges ---
        # Here we apply the mask on the final continuous profile using the original q grid.
        if mask_ranges is not None:
            for (mask_low, mask_high) in mask_ranges:
                overall_iq_cont[(q_cont >= mask_low) & (q_cont <= mask_high)] = 0.0

        return {
            'q': q_cont.numpy(),
            'iq': overall_iq_cont.numpy(),
            'q_disc': [qd.numpy() for qd in q_disc_list],
            'iq_disc': [idisc.numpy() for idisc in iq_disc_list]
        }

    except Exception as e:
        if debug:
            print(f"General error in processing {structure_name}: {e}")
        return None

def get_atomic_props_block(composition, oxi=False):
    noble_vdw_radii = {
        "He": 1.40,
        "Ne": 1.54,
        "Ar": 1.88,
        "Kr": 2.02,
        "Xe": 2.16,
        "Rn": 2.20,
    }

    allen_electronegativity = {
        "He": 4.16,
        "Ne": 4.79,
        "Ar": 3.24,
    }

    def _format(val):
        return f"{float(val): .4f}"

    def _format_X(elem):
        if math.isnan(elem.X) and str(elem) in allen_electronegativity:
            return allen_electronegativity[str(elem)]
        return _format(elem.X)

    def _format_radius(elem):
        if elem.atomic_radius is None and str(elem) in noble_vdw_radii:
            return noble_vdw_radii[str(elem)]
        return _format(elem.atomic_radius)

    props = {str(el): (_format_X(el), _format_radius(el), _format(el.average_ionic_radius))
             for el in sorted(composition.elements)}

    data = {}
    data["_atom_type_symbol"] = list(props)
    data["_atom_type_electronegativity"] = [v[0] for v in props.values()]
    data["_atom_type_radius"] = [v[1] for v in props.values()]
    # use the average ionic radius
    data["_atom_type_ionic_radius"] = [v[2] for v in props.values()]

    loop_vals = [
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",
        "_atom_type_ionic_radius"
    ]

    if oxi:
        symbol_to_oxinum = {str(el): (float(el.oxi_state), _format(el.ionic_radius)) for el in sorted(composition.elements)}
        data["_atom_type_oxidation_number"] = [v[0] for v in symbol_to_oxinum.values()]
        # if we know the oxidation state of the element, use the ionic radius for the given oxidation state
        data["_atom_type_ionic_radius"] = [v[1] for v in symbol_to_oxinum.values()]
        loop_vals.append("_atom_type_oxidation_number")

    loops = [loop_vals]

    return str(CifBlock(data, loops, "")).replace("data_\n", "")

def extract_species(cif_str):
    return list(set(Composition(extract_formula_nonreduced(cif_str)).as_dict().keys()))

def extract_composition(cif_string):
    return Composition(extract_formula_nonreduced(cif_string)).as_dict()

def add_atomic_props_block(cif_str, oxi=False):
    comp = Composition(extract_formula_nonreduced(cif_str))

    block = get_atomic_props_block(composition=comp, oxi=oxi)

    # the hypothesis is that the atomic properties should be the first thing
    #  that the model must learn to associate with the composition, since
    #  they will determine so much of what follows in the file
    pattern = r"_symmetry_space_group_name_H-M"
    match = re.search(pattern, cif_str)

    if match:
        start_pos = match.start()
        modified_cif = cif_str[:start_pos] + block + "\n" + cif_str[start_pos:]
        return modified_cif
    else:
        raise Exception(f"Pattern not found: {cif_str}")

def replace_symmetry_loop_with_P1(cif_str):
    start = cif_str.find("_symmetry_equiv_pos_site_id")
    end = cif_str.find("loop_", start+1)

    replacement = """_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1  'x, y, z'\n"""

    return cif_str.replace(cif_str[start:end], replacement)

def reinstate_symmetry_loop(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    try:
        ops = [op.as_xyz_string() for op in symmops]
    except:
        ops = [op.as_xyz_str() for op in symmops]
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops

    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    pattern = r"(loop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1  'x, y, z')"
    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated

def remove_cif_header(cif_str):
    lines = cif_str.split('\n')
    cif_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            cif_lines.append(line)

    cif_str = '\n'.join(cif_lines)
    return cif_str

OXI_LOOP_PATTERN = r'loop_[^l]*(?:l(?!oop_)[^l]*)*_atom_type_oxidation_number[^l]*(?:l(?!oop_)[^l]*)*'
OXI_STATE_PATTERN = r'(\n\s+)([A-Za-z]+)[\d.+-]*'

# Regular expression to match occupancy values
OCCU_PATTERN = re.compile(
        r'(_atom_site_type_symbol[\s\S]+?_atom_site_occupancy\s+)((?:\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\.\d+\s*)+)', re.MULTILINE)

def remove_oxidation_loop(cif_str):
    cif_str = re.sub(OXI_LOOP_PATTERN, '', cif_str)
    cif_str = re.sub(OXI_STATE_PATTERN, r'\1\2', cif_str)
    return cif_str

def format_occupancies(cif_str, decimal_places=4):

    # Search for the occupancy block
    occupancy_block_match = OCCU_PATTERN.search(cif_str)
    
    if occupancy_block_match:
        # Extract header and atoms
        header = occupancy_block_match.group(1)
        atoms = occupancy_block_match.group(2).strip().split('\n')
        
        # Check occupancies and format them
        formatted_atoms = []
        occupancies = []
        for atom in atoms:
            parts = atom.split()
            occupancy = float(parts[-1])
            occupancies.append(occupancy)
            formatted_occupancy = "{:.{}f}".format(occupancy,decimal_places)
            formatted_atoms.append('  '.join(parts[:-1]) + f"  {formatted_occupancy}")

        formatted_block = header + '\n  '.join(formatted_atoms)
        cif_str = OCCU_PATTERN.sub(formatted_block, cif_str)

    return cif_str

def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)

def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")

def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    raise Exception(f"could not find {prop} in:\n{cif_str}")

def extract_data_formula(cif_str):
    #match = re.search(r"data_([A-Za-z0-9]+)\n", cif_str)
    match = re.search(r"data_([A-Za-z0-9.()]+)\n", cif_str) # Including paranthesis
    if match:
        return match.group(1)
    raise Exception(f"could not find data_ in:\n{cif_str}")

def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum value from:\n{cif_str}")

def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1)
        chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

        modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)

        return modified_cif
    else:
        raise Exception(f"Chemical formula not found {cif_str}")

def round_numbers(cif_str, decimal_places=4):
    # Pattern to match a floating point number in the CIF file
    # It also matches numbers in scientific notation
    # pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"
    pattern = r"[-+]?\d*\.?\d+([eE][-+]?\d+)?" # Including occupancies

    # Function to round the numbers
    def round_number(match):
        number_str = match.group()
        number = float(number_str)
        # Check if number of digits after decimal point is less than 'decimal_places'
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        rounded = round(number, decimal_places)
        return format(rounded, '.{}f'.format(decimal_places))

    # Replace all occurrences of the pattern using a regex sub operation
    cif_string_rounded = re.sub(pattern, round_number, cif_str)

    return cif_string_rounded

def get_unit_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha_rad = math.radians(alpha_deg)
    beta_rad = math.radians(beta_deg)
    gamma_rad = math.radians(gamma_deg)

    volume = (a * b * c * math.sqrt(1 - math.cos(alpha_rad) ** 2 - math.cos(beta_rad) ** 2 - math.cos(gamma_rad) ** 2 +
                                    2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)))

    return volume

def extract_volume(cif_str):
    return extract_numeric_property(cif_str, "_cell_volume")


def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            j = connected_site_info['site_index']
            if i == j:  # skip if they're the same site
                continue
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

            is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

            electronegativity_diff = abs(site.specie.X - connected_site.specie.X)
            """
            According to the Pauling scale, when the electronegativity difference 
            between two bonded atoms is less than 1.7, the bond can be considered 
            to have predominantly covalent character, while a difference greater 
            than or equal to 1.7 indicates that the bond has significant ionic 
            character.
            """
            if electronegativity_diff >= 1.7:
                # use ionic radii
                if site.specie.X < connected_site.specie.X:
                    expected_length = site.specie.average_cationic_radius + connected_site.specie.average_anionic_radius
                else:
                    expected_length = site.specie.average_anionic_radius + connected_site.specie.average_cationic_radius
            else:
                expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius

            bond_ratio = bond_length / expected_length

            # penalize bond lengths that are too short or too long;
            #  check if bond involves hydrogen and adjust tolerance accordingly
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    normalized_score = score / bond_count

    return normalized_score

def is_space_group_consistent(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")
    try:
        parser = CifParser.from_string(cif_str)
    except:
        parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    # Extract the stated space group from the CIF file
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']

    # Analyze the symmetry of the structure
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()

    # Check if the detected space group matches the stated space group
    is_match = stated_space_group.strip() == detected_space_group.strip()

    return is_match

def is_formula_consistent(cif_str):
    try:
        parser = CifParser.from_string(cif_str)
    except:
        parser = CifParser.from_str(cif_str)

    cif_data = parser.as_dict()

    formula_data = Composition(extract_data_formula(cif_str))
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])

    return formula_data.reduced_formula == formula_sum.reduced_formula == formula_structural.reduced_formula

def is_atom_site_multiplicity_consistent(cif_str):
    # Parse the CIF string
    try:
        parser = CifParser.from_string(cif_str)
    except:
        parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    # Extract the chemical formula sum from the CIF data
    formula_sum = cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"]

    # Convert the formula sum into a dictionary
    expected_atoms = Composition(formula_sum).as_dict()

    # Count the atoms provided in the _atom_site_type_symbol section
    actual_atoms = {}
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):
                
                if atom_type in actual_atoms:
                    actual_atoms[atom_type] += int(multiplicity)
                else:
                    actual_atoms[atom_type] = int(multiplicity)

    return expected_atoms == actual_atoms

def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    
    try: 
        Structure.from_str(cif_str, fmt="cif")
    except:
        return False
    
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True

def is_valid(cif_str, bond_length_acceptability_cutoff=1.0):
    if not is_formula_consistent(cif_str):
        return False
    if not is_atom_site_multiplicity_consistent(cif_str):
        return False

    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score < bond_length_acceptability_cutoff:
         return False
    if not is_space_group_consistent(cif_str):
        return False
    return True

def evaluate_syntax_validity(cif_str, bond_length_acceptability_cutoff=1.0):
    validity_dict = {
        "formula": False,
        "site_multiplicity": False, 
        "bond_length": False, 
        "spacegroup": False,
    }
    if is_formula_consistent(cif_str):
        validity_dict["formula"] = True
    if is_atom_site_multiplicity_consistent(cif_str):
        validity_dict["site_multiplicity"] = True
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score >= bond_length_acceptability_cutoff:
        validity_dict["bond_length"] = True
    if is_space_group_consistent(cif_str):
        validity_dict["spacegroup"] = True
    
    return validity_dict

def space_group_symbol_to_number(symbol):
    # Dictionary mapping space group symbols to their corresponding numbers
    space_group_mapping = {
        
        # Triclinic
        "P1": 1, "P-1": 2, 

        # Monoclinic
        "P2": 3, "P2_1": 4, "C2": 5, "Pm": 6, "Pc": 7, "Cm": 8, "Cc": 9,
        "P2/m": 10, "P2_1/m": 11, "C2/m": 12, "P2/c": 13, "P2_1/c": 14, "C2/c": 15,
        "P222": 16, 

        # Orthorombic
        "P222_1": 17, "P2_12_12": 18, "P2_12_12_1": 19, "C222_1": 20, "C222": 21,
        "F222": 22, "I222": 23, "I2_12_12_1": 24, "Pmm2": 25, "Pmc2_1": 26, "Pcc2": 27,
        "Pma2": 28, "Pca2_1": 29, "Pnc2": 30, "Pmn2_1": 31, "Pba2": 32, "Pna2_1": 33,
        "Pnn2": 34, "Cmm2": 35, "Cmc2_1": 36, "Ccc2": 37, "Amm2": 38, "Aem2": 39,
        "Ama2": 40, "Aea2": 41, "Fmm2": 42, "Fdd2": 43, "Imm2": 44, "Iba2": 45,
        "Ima2": 46, "Pmmm": 47, "Pnnn": 48, "Pccm": 49, "Pban": 50, "Pmma": 51,
        "Pnna": 52, "Pmna": 53, "Pcca": 54, "Pbam": 55, "Pccn": 56, "Pbcm": 57,
        "Pnnm": 58, "Pmmn": 59, "Pbcn": 60, "Pbca": 61, "Pnma": 62, "Cmcm": 63,
        "Cmce": 64, "Cmmm": 65, "Cccm": 66, "Cmme": 67, "Ccce": 68, "Fmmm": 69,
        "Fddd": 70, "Immm": 71, "Ibam": 72, "Ibca": 73, "Imma": 74, 

        # Tetragonal
        "P4": 75,
        "P4_1": 76, "P4_2": 77, "P4_3": 78, "I4": 79, "I4_1": 80, "P-4": 81, "I-4": 82,
        "P4/m": 83, "P4_2/m": 84, "P4/n": 85, "P4_2/n": 86, "I4/m": 87, "I4_1/a": 88,
        "P422": 89, "P42_12": 90, "P4_122": 91, "P4_12_12": 92, "P4_222": 93, "P4_22_12": 94,
        "P4_322": 95, "P4_32_12": 96, "I422": 97, "I4_122": 98, "P4mm": 99, "P4bm": 100,
        "P4_2cm": 101, "P4_2nm": 102, "P4cc": 103, "P4nc": 104, "P4_2mc": 105, "P4_2bc": 106,
        "I4mm": 107, "I4cm": 108, "I4_1md": 109, "I4_1cd": 110, "P-42m": 111, "P-42c": 112,
        "P-42_1m": 113, "P-42_1c": 114, "P-4m2": 115, "P-4c2": 116, "P-4b2": 117, "P-4n2": 118,
        "I-4m2": 119, "I-4c2": 120, "I-42m": 121, "I-42d": 122, "P4/mmm": 123, "P4/mcc": 124,
        "P4/nbm": 125, "P4/nnc": 126, "P4/mbm": 127, "P4/mnc": 128, "P4/nmm": 129, "P4/ncc": 130,
        "P4_2/mmc": 131, "P4_2/mcm": 132, "P4_2/nbc": 133, "P4_2/nnm": 134, "P4_2/mbc": 135,
        "P4_2/mnm": 136, "P4_2/nmc": 137, "P4_2/ncm": 138, "I4/mmm": 139, "I4/mcm": 140,
        "I4_1/amd": 141, "I4_1/acd": 142, 

        # Trigonal
        "P3": 143, "P3_1": 144,
        "P3_2": 145, "R3": 146, "P-3": 147, "R-3": 148, "P312": 149, "P321": 150, "P3_112": 151,
        "P3_121": 152, "P3_212": 153, "P3_221": 154, "R32": 155, "P3m1": 156, "P31m": 157,
        "P3c1": 158, "P31c": 159, "R3m": 160, "R3c": 161, "P-31m": 162, "P-31c": 163,
        "P-3m1": 164, "P-3c1": 165, "R-3m": 166, "R-3c": 167,

        # Hexagonal
        "P6": 168, "P6_1": 169,
        "P6_5": 170, "P6_2": 171, "P6_4": 172, "P6_3": 173, "P-6": 174, "P6/m": 175,
        "P6_3/m": 176, "P622": 177, "P6_122": 178, "P6_522": 179, "P6_222": 180, "P6_422": 181,
        "P6_322": 182, "P6mm": 183, "P6cc": 184, "P6_3cm": 185, "P6_3mc": 186, "P-62m": 187,
        "P-62c": 188, "P-6m2": 189, "P-6c2": 190, "P6/mmm": 191, "P6/mcc": 192, "P6_3/mcm": 193,
        "P6_3/mmc": 194, 

        # Cubic
        "P23": 195, "F23": 196, "I23": 197, "P2_13": 198, "I2_13": 199,
        "Pm-3": 200, "Pn-3": 201, "Fm-3": 202, "Fd-3": 203, "Im-3": 204, "Pa-3": 205,
        "Ia-3": 206, "P432": 207, "P4_232": 208, "F432": 209, "F4_132": 210, "I432": 211,
        "P4_332": 212, "P4_132": 213, "I4_132": 214, "P-43m": 215, "F-43m": 216, "I-43m": 217,
        "P-43n": 218, "F-43c": 219, "I-43d": 220, "Pm-3m": 221, "Pn-3n": 222, "Pm-3n": 223,
        "Pn-3m": 224, "Fm-3m": 225, "Fm-3c": 226, "Fd-3m": 227, "Fd-3c": 228, "Im-3m": 229,
        "Ia-3d": 230
    }
    
    # Convert the space group symbol to the space group number
    return space_group_mapping.get(symbol, None)

def get_metrics(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    try:
        metrics = checkpoint['training_metrics']
    except:
        metrics = checkpoint['metrics']
    fname = '/'.join(ckpt_path.split('/')[:-1])
    return metrics, fname

# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py
# and from
# https://github.com/FrederikLizakJohansen/CrystaLLM/blob/2d130f9d561136a8745ab7568ebcbd69cdac913f/bin/benchmark_metrics.py
def get_rmsd(cif_string_sample, cif_string_gen, matcher):
    """ Presumes valid structures or exception is thrown """
    try:
        structure_sample = Structure.from_str(cif_string_sample, fmt="cif")
        structure_gen = Structure.from_str(cif_string_gen, fmt="cif")
        rmsd = matcher.get_rms_dist(structure_sample, structure_gen)
        if rmsd is not None:
            return rmsd[0] # Mean
        else:
            return None
    except Exception as e:
        print(e)
        return None

def plot_loss_curves(paths, ylog=True, xlog=False, xmin=None, xmax=None, ymin=None, ymax=None, offset=0.02, plot_metrics=True, figsize=(10, 5)):
    # Apply Seaborn style
    sns.set_theme(style="whitegrid")
    #plt.figure(figsize=figsize, dpi=150)
    fig, ax_loss = plt.subplots(figsize=figsize, dpi=150)
    text_positions = []

    ax_loss.set(xlabel='Epoch', ylabel='Cross-Entropy Loss')

    if xmin is not None or xmax is not None:
        ax_loss.set_xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
        ax_loss.set_ylim(ymin, ymax)
    if ylog:
        ax_loss.set_yscale('log')
    if xlog:
        ax_loss.set_xscale('log')
    
    # Seaborn grid
    ax_loss.grid(alpha=0.4, which="both")

    for path in paths:
        metrics, fname = get_metrics(path)
        legend_fname = '/'.join(fname.split("/")[-2:])
        
        # Convert lists to numpy arrays for plotting
        losses_train = np.array(metrics['train_losses'])
        losses_val = np.array(metrics['val_losses'])
        try:
            epochs = np.array(metrics['epoch_losses'])
        except:
            epochs = np.array(metrics['epochs'])
        
        # Train and validation loss plots
        p = sns.lineplot(x=epochs, y=losses_train, label=f'{legend_fname} [Train]', ax=ax_loss, linewidth=2)
        sns.lineplot(x=epochs, y=losses_val, label=f'{legend_fname} [Validation]', ax=ax_loss, linewidth=2, linestyle='--', color=p.get_lines()[-1].get_color())
        
        # Find the minimum value in losses_val and its corresponding epoch
        try:
            val_line_min = epochs[np.argmin(losses_val)]
            min_loss_val = np.min(losses_val)
        
            # Plot the dotted line
            ax_loss.plot([val_line_min, ax_loss.get_xlim()[1]], [min_loss_val, min_loss_val],
                        c=p.get_lines()[-1].get_color(), ls=':', alpha=1.0)

            # Adjust text position if overlapping
            text_x = ax_loss.get_xlim()[1]
            text_y = min_loss_val
        
            vert_align = 'bottom'
            for pos in text_positions:
                if abs(pos[1] - text_y) < offset:  # Check for overlap
                    vert_align = 'top'
                else:
                    vert_align = 'bottom'

            # Add text at the end of the dotted line
            ax_loss.text(text_x, text_y, f'{min_loss_val:.4f}', 
                    verticalalignment=vert_align, horizontalalignment='right', color=p.get_lines()[-1].get_color(),
                    fontsize=10)
            text_positions.append((text_x, text_y))
        except Exception as e:
            print(f"Error plotting validation loss for {legend_fname}: {e}")
            pass
     
    # Add the legend
    ax_loss.legend(fontsize=8)
    
    # Make the layout tight
    fig.tight_layout()

    # Show the plot
    plt.show()

