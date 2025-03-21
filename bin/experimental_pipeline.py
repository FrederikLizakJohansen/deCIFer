import os
import zipfile
from typing import Optional, List, Tuple, Union, Dict, Any
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import torch
from pymatgen.core import Structure
from pymatgen.core import Structure as PMGStructure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.groups import SpaceGroup

# Minimal imports from your modules (adjust paths as needed):
from bin.evaluate import load_model_from_checkpoint
from bin.train import TrainConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    pxrd_from_cif,
    replace_symmetry_loop_with_P1,
    extract_space_group_symbol,
    reinstate_symmetry_loop,
    space_group_to_crystal_system,
    space_group_symbol_to_number
)
from tqdm.auto import tqdm

from ase.visualize.plot import plot_atoms
from ase.data import colors, atomic_numbers


class DeciferPipeline:
    """
    A pipeline to preprocess experimental data, generate CIF structures using a trained model,
    and plot the results alongside PXRD predictions.
    """

    def __init__(self, model_path: str, zip_path: str, device: str = "cuda",
                 temperature: float = 1.0, max_new_tokens: int = 3000, results_output_folder='./') -> None:
        """
        Initialize the DeciferPipeline with the model, experimental data and default parameters.

        Parameters:
            model_path (str): Path to the trained model checkpoint.
            zip_path (str): Path to the ZIP file containing experimental data.
            device (str): Device to run the model on ("cuda" or "cpu").
            temperature (float): Sampling temperature for generation.
            max_new_tokens (int): Maximum number of tokens to generate.
        """
        # Set up periodic table layout as a 2D list
        self.periodic_table_layout: List[List[Optional[str]]] = [
            # Period 1
            ["H", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "He"],
            # Period 2
            ["Li", "Be", None, None, None, None, None, None, None, None, None, None, "B", "C", "N", "O", "F", "Ne"],
            # Period 3
            ["Na", "Mg", None, None, None, None, None, None, None, None, None, None, "Al", "Si", "P", "S", "Cl", "Ar"],
            # Period 4
            ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
            # Period 5
            ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
            # Period 6 Main Block
            ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", None],
            # Period 7 Main Block
            ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", None],
            # Break row (empty)
            [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            # Lanthanides (Period 6)
            [None, None, None, "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", None],
            # Actinides (Period 7)
            [None, None, None, "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", None]
        ]
        # Flatten the periodic table into a list of active elements
        self.global_inactive_elements_list: List[str] = []
        for row in self.periodic_table_layout:
            for el in row:
                if el is not None:
                    self.global_inactive_elements_list.append(el)

        # Initialize tokenizer and related token variables
        self.TOKENIZER: Tokenizer = Tokenizer()
        self.VOCAB_SIZE: int = self.TOKENIZER.vocab_size
        self.START_ID: int = self.TOKENIZER.token_to_id["data_"]
        self.PADDING_ID: int = self.TOKENIZER.padding_id
        self.NEWLINE_ID: int = self.TOKENIZER.token_to_id["\n"]
        self.SPACEGROUP_ID: int = self.TOKENIZER.token_to_id["_symmetry_space_group_name_H-M"]
        self.DECODE = self.TOKENIZER.decode
        self.ENCODE = self.TOKENIZER.encode
        self.TOKENIZE = self.TOKENIZER.tokenize_cif

        # Model and generation parameters
        self.device: str = device
        self.model: Optional[torch.nn.Module] = self.load_custom_model(model_path)
        self.temperature: float = temperature
        self.max_new_tokens: int = max_new_tokens

        # Setup folder
        self.setup_folder(results_output_folder)

        # Setup experimental data and placeholders for processed data and results
        self.df_exp: pd.DataFrame = self.read_experimental_data(zip_path)
        self.df_processed: Optional[pd.DataFrame] = None
        self.results: Optional[Dict[str, Any]] = None
        self.exp_i: Optional[np.ndarray] = None
        self.exp_q: Optional[np.ndarray] = None
        self.raw_i: Optional[np.ndarray] = None
        self.raw_q: Optional[np.ndarray] = None

    def setup_folder(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.results_output_folder = path

    def prepare_target_data(
        self,
        target_file: str,
        background_file: Optional[str] = None,
        q_min_crop: float = 0.0,
        q_max_crop: float = 10.0,
        wavelength: Union[float, str] = 'CuKa',
        n_points: int = 1000
    ) -> None:
        """
        Prepares target experimental data by cropping and normalizing the signal.

        Parameters:
            target_file (str): File name of the target experimental data.
            background_file (Optional[str]): Optional background data file for subtraction.
            q_min_crop (float): Minimum Q value for cropping.
            q_max_crop (float): Maximum Q value for cropping.
            wavelength (Union[float, str]): Wavelength used in the experiment.
            n_points (int): Number of points to standardize onto.
        """
        # Set preparation configuration
        self.prep_config: Dict[str, Union[str, float, int]] = {
            "target_file": target_file,
            "background_file": background_file,
            "q_min_crop": q_min_crop,
            "q_max_crop": q_max_crop,
            "wavelength": wavelength,
            "n_points": n_points,
        }
        # Preprocess and update the processed dataframe and signal arrays
        self.df_processed = self.preprocess_generic(**self.prep_config)
        self.exp_q = self.df_processed['Q'].values
        self.exp_i = self.df_processed['intensity_crop_norm'].values
        self.raw_q = self.df_processed['Q'].values
        self.raw_i = self.df_processed['intensity_norm']

    def read_experimental_data(self, zip_path: str) -> pd.DataFrame:
        """
        Reads and combines all .xy or .xye files in the ZIP archive,
        maps them to their compositions, and returns a single DataFrame.

        Parameters:
            zip_path (str): Path to the ZIP file containing experimental data.

        Returns:
            pd.DataFrame: Combined experimental data.
        """
        # Read composition information from an Excel file within the ZIP
        #with zipfile.ZipFile(zip_path, 'r') as z:
            #with z.open('experimental_data/Jens/Compositions.xlsx') as f:
            #    df_jens_comp = pd.read_excel(f, engine='openpyxl')
        #df_jens_comp['Name_lower'] = df_jens_comp['Name'].str.lower()

        #def get_jens_composition(fname: str) -> Optional[str]:
        #    """Infer composition from file name based on Jens' data."""
        #    fname_lower = fname.lower()
        #    for _, row in df_jens_comp.iterrows():
        #        if row['Name_lower'] in fname_lower:
        #            return row['Composition']
        #    return None

        frames: List[pd.DataFrame] = []
        # Process each file in the ZIP archive
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                fn = info.filename
                if not (fn.endswith('.xy') or fn.endswith('.xye')):
                    continue

                with z.open(fn) as f:
                    lines = f.read().decode('utf-8').splitlines()

                folder = os.path.basename(os.path.dirname(fn))
                base_name = os.path.basename(fn)
                # Infer composition based on the folder and file name
                # if folder.lower() == 'jens':
                #     comp_str = get_jens_composition(base_name)
                # elif folder.lower() == 'laura_irox':
                #     if 'irox' in base_name.lower():
                #         comp_str = 'IrOx'
                #     elif 'iro2' in base_name.lower():
                #         comp_str = 'IrO2'
                #     else:
                #         comp_str = 'Ir-based oxide'
                # elif folder.lower() == 'nicolas':
                #     subfolder = fn.split('/')[-2]
                #     if 'fcc pure' in subfolder.lower():
                #         comp_str = 'Pt (fcc)'
                #     elif 'fcc+fct' in subfolder.lower():
                #         comp_str = 'Pt (fcc+fct)'
                #     else:
                #         comp_str = 'Pt-based'
                # elif folder.lower() == 'rebecca_ceo2':
                #     comp_str = 'CeO2'
                # else:
                #comp_str = None

                records: List[Tuple[float, float, Optional[float]]] = []
                # Process each line in the file
                for line in lines:
                    parts = line.split()
                    if len(parts) == 2:
                        angle, intensity = parts
                        records.append((float(angle), float(intensity), None))
                    elif len(parts) == 3:
                        angle, intensity, error = parts
                        records.append((float(angle), float(intensity), float(error)))

                df_temp = pd.DataFrame(records, columns=['angle', 'intensity', 'error'])
                df_temp['source_file'] = base_name
                df_temp['source_folder'] = folder
                frames.append(df_temp)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def standardize_signal(
        self,
        df: pd.DataFrame,
        q_col: str,
        intensity_col: str,
        q_min: float = 0.0,
        q_max: float = 10.0,
        n_points: int = 1000
    ) -> pd.DataFrame:
        """
        Standardizes the signal by interpolating onto a common Q grid.

        Parameters:
            df (pd.DataFrame): DataFrame containing the signal.
            q_col (str): Column name for Q values.
            intensity_col (str): Column name for intensity values.
            q_min (float): Minimum Q value.
            q_max (float): Maximum Q value.
            n_points (int): Number of points in the standardized grid.

        Returns:
            pd.DataFrame: DataFrame with standardized Q and intensity values.
        """
        df = df.drop_duplicates(subset=q_col).sort_values(by=q_col)
        q_new = np.linspace(q_min, q_max, n_points)
        i_new = np.interp(q_new, df[q_col], df[intensity_col])
        return pd.DataFrame({q_col: q_new, intensity_col: i_new})

    def load_custom_model(self, model_path: str, device: Optional[str] = None) -> torch.nn.Module:
        """
        Loads a custom model from a checkpoint.

        Parameters:
            model_path (str): Path to the model checkpoint.
            device (Optional[str]): Device to load the model on (defaults to self.device).

        Returns:
            torch.nn.Module: The loaded model.
        """
        if device is None:
            device = self.device
        self.model = load_model_from_checkpoint(model_path, device=device)
        return self.model

    def fix_symmetry_in_cif(self, cif_string: str) -> str:
        """
        Fixes the symmetry of a CIF string.

        Parameters:
            cif_string (str): The raw CIF string.

        Returns:
            str: The CIF string with corrected symmetry.
        """
        # Replace the symmetry loop with a P1 structure
        c = replace_symmetry_loop_with_P1(cif_string)
        sg = extract_space_group_symbol(c)
        # Reinstate the symmetry loop if the space group is not P 1
        return reinstate_symmetry_loop(c, sg) if sg != "P 1" else c

    def run_decifer_generation(
        self,
        cond_array: Union[torch.Tensor, np.ndarray],
        composition: Optional[str] = None,
        composition_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        spacegroup: Optional[str] = None,
        do_plot: bool = False,
        exclusive_elements: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        crystal_systems: Optional[List[str]] = None
    ) -> Optional[Tuple[str, Structure]]:
        """
        Generates a CIF string from the provided condition vector and returns the CIF and its corresponding structure.

        Parameters:
            cond_array (Union[torch.Tensor, np.ndarray]): Conditioning array for generation.
            composition (Optional[str]): Optional composition string.
            spacegroup (Optional[str]): Optional space group string.
            do_plot (bool): Flag to indicate whether to plot (unused in current implementation).
            exclusive_elements (Optional[List[str]]): List of elements to exclude.
            temperature (Optional[float]): Temperature for generation.
            max_new_tokens (Optional[int]): Maximum tokens to generate.
            crystal_systems (Optional[List[str]]): List of target crystal systems.

        Returns:
            Optional[Tuple[str, Structure]]: A tuple of the fixed CIF string and the corresponding Structure,
                                               or None if generation fails.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model using load_custom_model().")
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Ensure condition array is a torch.Tensor and adjust dimensions
        if not isinstance(cond_array, torch.Tensor):
            cond_array = torch.tensor(cond_array)
        cond_array = cond_array.unsqueeze(0).to(self.model.device).float()

        # Determine inactive elements if exclusive elements are provided
        if exclusive_elements is not None:
            inactive_elements_list = [el for el in self.global_inactive_elements_list if el not in exclusive_elements]
        else:
            inactive_elements_list = None

        # Determine active space groups if crystal systems are provided
        if crystal_systems is not None:
            active_spacegroups: List[str] = []
            for cs in crystal_systems:
                active_spacegroups.extend(self.get_space_group_symbols(cs, include=True))
        else:
            active_spacegroups = None

        # Create prompt tokens for generation
        prompt = torch.tensor([self.START_ID]).unsqueeze(0).to(self.model.device)
        if composition:
            comp_str = f"data_{composition}\n"
            c_tokens = self.ENCODE(self.TOKENIZE(comp_str))
            prompt = torch.tensor(c_tokens).unsqueeze(0).to(self.model.device)

        # Generate new tokens using the model's custom generate function
        out = self.model.generate_custom(
            idx=prompt,
            max_new_tokens=max_new_tokens,
            cond_vec=cond_array,
            start_indices_batch=[[0]],
            composition_string=composition,
            composition_ranges=composition_ranges,
            spacegroup_string=spacegroup,
            exclude_elements=inactive_elements_list,
            temperature=temperature,
            disable_pbar=False,
            include_spacegroups=active_spacegroups,
        ).cpu().numpy()
        cif_raw: str = self.DECODE(out[0])

        try:
            # Fix the symmetry in the generated CIF string and convert it to a Structure object
            cif_fixed = self.fix_symmetry_in_cif(cif_raw)
            structure = Structure.from_str(cif_fixed, fmt="cif")
            return cif_fixed, structure
        except Exception as e:
            return None

    def preprocess_generic(
        self,
        target_file: str,
        wavelength: Union[float, str] = 0.25448,
        q_min_crop: float = 1.0,
        q_max_crop: float = 8.0,
        n_points: int = 1000,
        background_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Preprocesses experimental data by performing background subtraction, normalization,
        baseline correction, and standardization onto a common Q grid.

        Parameters:
            target_file (str): Target file name from the experimental data.
            wavelength (Union[float, str]): Wavelength for conversion from angle to Q.
            q_min_crop (float): Minimum Q value for cropping.
            q_max_crop (float): Maximum Q value for cropping.
            n_points (int): Number of points for standardization.
            background_file (Optional[str]): Optional background file for subtraction.

        Returns:
            pd.DataFrame: The preprocessed experimental data.
        """
        # 1. Filter for sample and compute Q
        df_sel = self.df_exp[self.df_exp['source_file'].str.lower() == target_file.lower()].copy()
        theta_rad = np.radians(df_sel["angle"] / 2.0)
        df_sel["Q"] = (4.0 * np.pi / float(wavelength)) * np.sin(theta_rad)

        # 2. Background subtraction with scaling
        if background_file is not None:
            bg_df = self.df_exp[self.df_exp['source_file'].str.lower() == background_file.lower()].copy()
            theta_rad_bg = np.radians(bg_df["angle"] / 2.0)
            bg_df["Q"] = (4.0 * np.pi / float(wavelength)) * np.sin(theta_rad_bg)
            bg_df.sort_values(by="Q", inplace=True)
            df_sel["background_intensity"] = np.interp(df_sel["Q"], bg_df["Q"], bg_df["intensity"])
            valid = df_sel["background_intensity"] > 0
            if valid.any():
                s = (df_sel.loc[valid, "intensity"] / df_sel.loc[valid, "background_intensity"]).min()
            else:
                s = 1.0
            df_sel["scaled_background"] = s * df_sel["background_intensity"]
            df_sel["intensity_bg"] = df_sel["intensity"] - df_sel["scaled_background"]
        else:
            df_sel["intensity_bg"] = df_sel["intensity"]

        # 3. Full signal normalization
        maxI = df_sel["intensity_bg"].max(skipna=True)
        df_sel["intensity_norm"] = df_sel["intensity_bg"] / maxI

        # 4. Crop for baseline correction
        df_crop_orig = df_sel[(df_sel["Q"] >= q_min_crop) & (df_sel["Q"] <= q_max_crop)].copy()
        baseline = df_crop_orig["intensity_bg"].min(skipna=True)
        df_crop_orig["intensity_corrected"] = df_crop_orig["intensity_bg"] - baseline
        df_endpoints = pd.DataFrame({"Q": [0.0, 10.0], "intensity_corrected": [0, 0]})
        df_crop = pd.concat([df_crop_orig[["Q", "intensity_corrected"]], df_endpoints], ignore_index=True)
        df_crop.sort_values(by="Q", inplace=True)

        # 5. Normalize the cropped, baseline-corrected signal
        max_val = df_crop_orig["intensity_corrected"].max(skipna=True)
        df_crop["intensity_normalized"] = df_crop["intensity_corrected"] / max_val

        # 6. Standardize signals onto a common Q grid
        Q_std = np.linspace(0, 10, n_points)
        df_final = pd.DataFrame({"Q": Q_std})
        df_final["intensity_original"] = np.interp(Q_std, df_sel["Q"], df_sel["intensity"])
        df_final["intensity_bg_subtracted"] = np.interp(Q_std, df_sel["Q"], df_sel["intensity_bg"])
        df_final["intensity_norm"] = np.interp(Q_std, df_sel["Q"], df_sel["intensity_norm"])
        df_final["intensity_crop_norm"] = np.interp(Q_std, df_crop["Q"], df_crop["intensity_normalized"])

        # 7. Include background signals in the output (if provided)
        if background_file is not None:
            df_final["intensity_bg"] = np.interp(Q_std, df_sel["Q"], df_sel["background_intensity"])
            df_final["intensity_scaled_bg"] = np.interp(Q_std, df_sel["Q"], df_sel["scaled_background"])
        return df_final

    def plot_unit_cell_with_boundaries(
        self,
        structure: Structure,
        ax: Optional[plt.Axes] = None,
        tol: float = 1e-5,
        radii: float = 0.8,
        rotation: Union[str, Tuple[str, ...]] = ('45x, -15y, 90z'),
        offset: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[plt.Axes, PMGStructure]:
        """
        Plots the unit cell with periodic boundaries using ASE and pymatgen.

        Parameters:
            structure (Structure): The pymatgen Structure to plot.
            ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, a new figure and axes are created.
            tol (float): Tolerance for duplicate fractional coordinates.
            radii (float): Radius for the atoms in the plot.
            rotation (Union[str, Tuple[str, ...]]): Rotation applied to the structure.
            offset (Tuple[float, float, float]): Offset applied to the structure.

        Returns:
            Tuple[plt.Axes, PMGStructure]: The axes and the discrete structure created for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Define translation vectors for periodic images
        translation_vectors = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ]

        all_species: List[str] = []
        all_coords: List[np.ndarray] = []

        # Loop over each translation vector to create periodic images
        for tv in translation_vectors:
            tv_cart = structure.lattice.get_cartesian_coords(tv)
            for site in structure:
                if tv == [0, 0, 0]:
                    all_species.append(site.species_string)
                    all_coords.append(site.coords)
                else:
                    # Only add sites that satisfy the fractional coordinate condition
                    if all(site.frac_coords[i] < tol for i, shift in enumerate(tv) if shift == 1):
                        all_species.append(site.species_string)
                        all_coords.append(site.coords + tv_cart)

        # Create a discrete structure using pymatgen
        discrete_structure = PMGStructure(
            lattice=structure.lattice.matrix,
            species=all_species,
            coords=np.array(all_coords),
            coords_are_cartesian=True
        )

        # Convert the structure to ASE atoms and disable periodic boundary conditions for plotting
        ase_atoms = AseAtomsAdaptor.get_atoms(discrete_structure)
        ase_atoms.set_pbc([False, False, False])
        plot_atoms(ase_atoms, ax, radii=radii, show_unit_cell=True, rotation=rotation, offset=offset)

        return ax, discrete_structure

    def plot_pxrd_and_structure(
        self,
        size_estimate: float,
        results: Optional[Dict[str, Any]] = None,
        vertical_lines: Optional[List[float]] = None,
        peak_scaling: float = 1.0,
        pred_marker_size: float = 3.0,
        atom_radii: float = 0.8,
        atom_tol: float = 1e-5,
        atom_offset: Tuple[float, float, float] = (0, 0, 0),
        atom_rotation: Union[str, Tuple[str, ...]] = ('45x, -15y, 90z'),
        atom_legend_radius: float = 1.2,
        struc_offset_x: float = 0.0,
        struc_offset_y: float = 0.25,
        struc_scale_width: float = 1.0,
        struc_scale_height: float = 1.0,
        struc_axis_off: bool = True,
        struc_add_axis_x: float = 2.0,
        struc_add_axis_y: float = 2.0,
        figsize: Tuple[float, float] = (10, 2),
        dpi: int = 300,
        base_fwhm: float = 0.05,
        complexity_weight: float = 0.1,
        show_estimation: bool = True,
        show_second_best: bool = False,
        second_best_threshold: float = 0.9,
    ) -> None:
        """
        Plots the experimental PXRD data and the predicted PXRD along with the corresponding structure.

        Parameters:
            size_estimate (float): Estimated particle size used for PXRD calculation.
            results (Optional[Dict[str, Any]]): Results dictionary from generation trials.
            vertical_lines (Optional[List[float]]): List of Q-values to draw vertical lines.
            peak_scaling (float): Scaling factor for peak intensities.
            pred_marker_size (float): Marker size for the predicted peaks.
            atom_radii (float): Radii used for atoms in structure plot.
            atom_tol (float): Tolerance for atom position filtering.
            atom_offset (Tuple[float, float, float]): Offset for structure plot.
            atom_rotation (Union[str, Tuple[str, ...]]): Rotation for the structure plot.
            atom_legend_radius (float): Radius for the atom legend circles.
            struc_offset_x (float): X offset for the inset structure plot.
            struc_offset_y (float): Y offset for the inset structure plot.
            struc_scale_width (float): Width scale for the inset structure plot.
            struc_scale_height (float): Height scale for the inset structure plot.
            struc_axis_off (bool): Flag to turn off axes for the inset plot.
            struc_add_axis_x (float): Additional x-axis margin for the inset plot.
            struc_add_axis_y (float): Additional y-axis margin for the inset plot.
            figsize (Tuple[float, float]): Figure size.
            dpi (int): Dots per inch for the figure.
            base_fwhm (float): Base full-width at half maximum for PXRD calculation.
            complexity_weight (float): Weight to balance ranking score based on complexity.
        """
        # Use provided results or the stored self.results
        results = results or self.results
        if results is None:
            raise ValueError("No results present. Provide a valid `results` list.")

        # First pass: Determine the best result
        best_result = None
        best_ranking_score = float("inf")
        best_rwp = float("inf")
        best_pxrd = None
        
        pbar_best = tqdm(total=len(results["gens"]), desc='Finding best structure...', leave=False)
        for res in results["gens"]:
            # Compute the PXRD from the generated CIF
            pxrd = pxrd_from_cif(res["cif_str"], base_fwhm=base_fwhm, particle_size=size_estimate)
            i_pred_interpolated = np.interp(self.exp_q, pxrd["q"], pxrd["iq"])
            rwp = np.sqrt(np.sum(np.square(self.exp_i - i_pred_interpolated)) / np.sum(np.square(self.exp_i)))
            n_peaks = len(pxrd["q_disc"][0])
            ranking_score = rwp + complexity_weight * n_peaks
            if ranking_score < best_ranking_score:
                best_rwp = rwp
                best_ranking_score = ranking_score
                best_result = res
                best_pxrd = pxrd 
            pbar_best.update(1)
        pbar_best.close()

        if best_result is None:
            print("No successful generations...")
            return

        from difflib import SequenceMatcher
        
        def is_approximately_same(str1: str, str2: str, threshold: float = 0.9) -> bool:
            """
            Returns True if the similarity ratio between str1 and str2 is greater than or equal to the threshold.
            """
            similarity = SequenceMatcher(None, str1, str2).ratio()
            return similarity >= threshold
        
        # Second pass: Determine the second best result that is distinct from the best sample
        second_best_result = None
        second_best_ranking_score = float("inf")
        second_best_rwp = float("inf")
        second_best_pxrd = None
        
        pbar_second_best = tqdm(total=len(results["gens"]), desc='Finding second best structure...', leave=False)
        for res in results["gens"]:
            # Use approximate string matching to skip candidates too similar to the best sample
            if is_approximately_same(res["cif_str"], best_result["cif_str"], threshold=second_best_threshold):
                pbar_second_best.update(1)
                continue
        
            # Compute the PXRD from the generated CIF for the candidate
            pxrd = pxrd_from_cif(res["cif_str"], base_fwhm=base_fwhm, particle_size=size_estimate)
            i_pred_interpolated = np.interp(self.exp_q, pxrd["q"], pxrd["iq"])
            rwp = np.sqrt(np.sum(np.square(self.exp_i - i_pred_interpolated)) / np.sum(np.square(self.exp_i)))
            n_peaks = len(pxrd["q_disc"][0])
            ranking_score = rwp + complexity_weight * n_peaks
        
            if ranking_score < second_best_ranking_score:
                second_best_rwp = rwp
                second_best_ranking_score = ranking_score
                second_best_result = res
                second_best_pxrd = pxrd
            pbar_second_best.update(1)
        pbar_second_best.close()

        # Plot experimental and predicted PXRD data
        fig, ax_data = plt.subplots(figsize=figsize, dpi=dpi)

        c_raw = "grey"
        c_stand = "k"
        c_pred_cont = "C2"
        c_pred = "C3"
        c_pred_2 = "C4"

        ax_data.plot(self.raw_q, self.raw_i * peak_scaling, label='Raw (max-norm)', color=c_raw, ls='--', lw=1, alpha=0.5)
        ax_data.plot(self.exp_q, self.exp_i * peak_scaling, label='Standardized', color=c_stand, ls='-', lw=1)
        if show_estimation:
            ax_data.plot(best_pxrd["q"], best_pxrd["iq"] * peak_scaling, label='Estimated', color=c_pred_cont, ls='-', lw=1)

        stem = ax_data.stem(np.array(best_pxrd["q_disc"][0]),
                            np.array(best_pxrd["iq_disc"][0]) / 100 * peak_scaling,
                            linefmt=f'{c_pred}-', markerfmt=f'{c_pred}o', basefmt=' ',
                            label='best prediction, $r_{\\mathrm{wp}}$/r_s:' + f'{best_rwp:1.2f}/' + f'{best_ranking_score:1.2f}')
        stem.markerline.set(markersize=pred_marker_size, markerfacecolor='white', markeredgecolor=c_pred, markeredgewidth=1.0)
        stem.markerline.set_xdata(np.array(best_pxrd["q_disc"]))

        if show_second_best:
            if second_best_pxrd is not None:
                stem = ax_data.stem(np.array(second_best_pxrd["q_disc"][0]),
                                    np.array(second_best_pxrd["iq_disc"][0]) / 100 * peak_scaling,
                                    linefmt=f'{c_pred_2}-', markerfmt=f'{c_pred_2}o', basefmt=' ',
                                    label='second_best prediction, $r_{\\mathrm{wp}}$/r_s:' + f'{second_best_rwp:1.2f}/' + f'{second_best_ranking_score:1.2f}')
                stem.markerline.set(markersize=pred_marker_size, markerfacecolor='white', markeredgecolor=c_pred_2, markeredgewidth=1.0)
                stem.markerline.set_xdata(np.array(second_best_pxrd["q_disc"]))
            else:
                print("No second best structure found...")

        # Compute and plot the average prediction (log-scaled)
        predictions_array = np.array(
            [pxrd_from_cif(res["cif_str"], base_fwhm=0.0125)["iq"] for res in results["gens"]]
        ).mean(axis=0).reshape(1, -1)
        predictions_array = np.log(predictions_array / (np.max(predictions_array) + 1e-16))
        ax_data.imshow(predictions_array, aspect='auto',
                       extent=[self.exp_q.min(), self.exp_q.max(), -0.05, 0],
                       origin='lower', cmap='viridis')
        legend_elements = [Patch(facecolor="grey", edgecolor="none", label="Peak Distribution")]
        ax_data.legend(handles=ax_data.get_legend_handles_labels()[0] + legend_elements)

        ax_data.grid(alpha=0.2, axis='x')
        ax_data.set(
            yticklabels=[],
            yticks=[],
            xticks=np.arange(0, 10, 1),
            xticklabels=np.arange(0, 10, 1),
            xlabel=r"$Q_{\;[Å^{-1}]}$",
            ylabel=r"$I(Q)_{\;[a.u.]}$",
            ylim=(-0.05, 1.05),
            xlim=(0, 9.99),
        )

        if vertical_lines is not None:
            for vline in vertical_lines:
                ax_data.axvline(x=vline, ymin=0, ls=':', lw=1, color='k')

        # Create an inset axes for the structure plot
        ax_struc = ax_data.inset_axes([struc_offset_x, struc_offset_y, struc_scale_width, struc_scale_height])
        pos = ax_data.get_position()
        ax_struc.set_position([pos.x0 + struc_offset_x, pos.y0 + struc_offset_y,
                               pos.width * struc_scale_width, pos.height * struc_scale_height])
        self.plot_unit_cell_with_boundaries(best_result["struct"], ax=ax_struc, radii=atom_radii,
                                            rotation=atom_rotation, offset=atom_offset, tol=atom_tol)
        max_coord = best_result["struct"].cart_coords.max()

        ax_struc.set_ylim(-struc_add_axis_y, max_coord * 3)
        ax_struc.set_xlim(-struc_add_axis_x, max_coord + struc_add_axis_x)
        if struc_axis_off:
            ax_struc.axis('off')

        # Create legend for the atomic species in the structure plot
        unique_species = sorted(set([site.species_string for site in best_result["struct"]]))
        n_species = len(unique_species)
        x_min, _ = ax_struc.get_xlim()
        legend_y = ax_struc.get_ylim()[0] + 2.0
        x_positions = np.linspace(max_coord - n_species * atom_legend_radius - atom_legend_radius,
                                  max_coord - atom_legend_radius, n_species)

        for x, species in zip(x_positions, unique_species):
            try:
                atom_color = colors.jmol_colors[atomic_numbers[species]]
            except KeyError:
                atom_color = 'black'
            circ = Circle((x, legend_y), radius=atom_legend_radius, edgecolor='black',
                          facecolor=atom_color, lw=1)
            ax_struc.add_patch(circ)
            ax_struc.text(x, legend_y, species, color='black', ha='center', va='center', fontsize=7 * atom_legend_radius)

        fig.tight_layout()
        plt.show()

    def get_space_group_symbols(self, crystal_system: str, include: bool = True) -> List[str]:
        """
        Returns a list of space group symbols for the given crystal system.

        Parameters:
            crystal_system (str): The target crystal system.
            include (bool): If True, returns symbols matching the crystal system;
                            if False, returns symbols that do NOT match.

        Returns:
            List[str]: A list of space group symbols with '_sg' appended.
        """
        sg_symbols: List[str] = []
        for number in range(1, 231):
            try:
                sg = SpaceGroup.from_int_number(number)
                symbol = sg.symbol
                is_match = (space_group_to_crystal_system(space_group_symbol_to_number(symbol)) == crystal_system)
                if (include and is_match) or (not include and not is_match):
                    sg_symbols.append(symbol + '_sg')
            except Exception:
                continue
        return sg_symbols

    def run_experiment_protocol(
        self,
        n_trials: int = 1,
        composition: Optional[str] = None,
        composition_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        spacegroup: Optional[str] = None,
        exclusive_elements: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        crystal_systems: Optional[List[str]] = None,
        save_to: Optional[str] = None,
        protocol_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs multiple generation trials and collects all results.

        Parameters:
            n_trials (int): Number of trials to run.
            composition (Optional[str]): Optional composition string.
            spacegroup (Optional[str]): Optional space group string.
            exclusive_elements (Optional[List[str]]): List of elements to exclude.
            temperature (Optional[float]): Temperature for generation.
            max_new_tokens (Optional[int]): Maximum tokens to generate.
            crystal_systems (Optional[List[str]]): List of target crystal systems.

        Returns:
            Dict[str, Any]: A dictionary containing generation results and experimental signals.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model using load_custom_model().")
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        results: Dict[str, Any] = {
            "gens": [],
            "generation_config": {
                "composition": composition,
                "n_trials": n_trials,
                "exclusive_elements": exclusive_elements,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "crystal_systems": crystal_systems,
                "spacegroup": spacegroup
            },
            "exp_q": self.exp_q.copy() if self.exp_q is not None else None,
            "exp_i": self.exp_i.copy() if self.exp_i is not None else None,
            "raw_q": self.raw_q.copy() if self.raw_q is not None else None,
            "raw_i": self.raw_i.copy() if self.raw_i is not None else None,
        }

        # Run the generation trials
        pbar_trials = tqdm(total=n_trials, desc=f'Running trials for protocol {protocol_name}', leave=True)
        for _ in range(n_trials):
            gen_out = self.run_decifer_generation(
                cond_array=self.exp_i,
                composition=composition,
                composition_ranges=composition_ranges,
                spacegroup=spacegroup,
                exclusive_elements=exclusive_elements,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                crystal_systems=crystal_systems,
            )
            if gen_out is not None:
                cif_str, struct = gen_out

                results["gens"].append({
                    "cif_str": cif_str,
                    "struct": struct,
                })
            pbar_trials.update(1)

        self.results = results
        if save_to:
            self.save_pickle(save_to)

    def save_pickle(self, output_file: str) -> None:
        """
        Saves the experimental data, generation results, and configuration to a pickle file.

        Parameters:
            output_file (str): Path to the output pickle file.
        """
        data_to_save = {
            "results": self.results,
            "exp_q": self.exp_q,
            "exp_i": self.exp_i,
            "raw_q": self.raw_q,
            "raw_i": self.raw_i,
            "prep_config": getattr(self, "prep_config", None)
        }
        with open(os.path.join(self.results_output_folder, output_file), "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Data saved to {os.path.join(self.results_output_folder, output_file)}")

    def load_pickle(self, input_file: str) -> None:
        """
        Loads the experimental data, generation results, and configuration from a pickle file.

        Parameters:
            input_file (str): Path to the input pickle file.
        """
        with open(input_file, "rb") as f:
            data_loaded = pickle.load(f)
        self.results = data_loaded.get("results", None)
        self.exp_q = data_loaded.get("exp_q", None)
        self.exp_i = data_loaded.get("exp_i", None)
        self.raw_q = data_loaded.get("raw_q", None)
        self.raw_i = data_loaded.get("raw_i", None)
        self.prep_config = data_loaded.get("prep_config", None)
        print(f"Data loaded from {input_file}")

