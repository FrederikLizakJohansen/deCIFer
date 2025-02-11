from typing import Optional, Tuple
from dash import Dash, html, dcc, Output, Input
import dash_uploader as du
import os
import plotly.graph_objs as go
from glob import glob
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
from pymatgen.io.cif import CifWriter
import torch
from pymatgen.core.structure import Structure
import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

from decifer.decifer_model import Decifer, DeciferConfig

# Init app
app = Dash(assets_folder=SETTINGS.ASSETS_PATH)
server = app.server

def generate_continuous_xrd_from_cif(
    cif_string,
    structure_name: str = 'null',
    wavelength: str = 'CuKa',
    qmin: float = 0.5,
    qmax: float = 8.0,
    qstep: float = 0.01,
    fwhm_range: Tuple[float, float] = (0.05, 0.05),
    eta_range: Tuple[float, float] = (0.5, 0.5),
    noise_range: Optional[Tuple[float, float]] = None,
    intensity_scale_range: Optional[Tuple[float, float]] = None,
    mask_prob: Optional[float] = None,
    debug: bool = False
):
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

# Define upload folder
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
du.configure_upload(app, UPLOAD_FOLDER)

from plotly.subplots import make_subplots  # Add this import at the top with your other imports

def generate_reference_pxrd(cif_string, height=100, width=525):
    out = generate_continuous_xrd_from_cif(cif_string, fwhm_range=(0.01, 0.01))

    if out is not None:
        fig = make_subplots(
            rows=1, cols=1,
        )
        fig.add_trace(
            go.Scatter(x=out['q'], y=out['iq'], mode='lines'),
            row=1, col=1
        )
        fig.update_layout(
            xaxis_title='Q [Å^-1]',
            yaxis_title='I(Q) [a.u.]',
            height=height,
            width=width,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        return fig
    else:
        return None

def generate_interactive_plot(cif_string):
    out = generate_continuous_xrd_from_cif(cif_string, fwhm_range=(0.01, 0.01))
    if out is not None:
        # Create a figure with 3 vertically stacked subplots sharing the x-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Continuous PXRD", "Discrete Peaks", "Combined PXRD")
        )
        # Subplot 1: Continuous PXRD pattern (line plot)
        fig.add_trace(
            go.Scatter(x=out['q'], y=out['iq'], mode='lines', name='Continuous PXRD'),
            row=1, col=1
        )
        # Subplot 2: Discrete peaks (markers)
        fig.add_trace(
            go.Scatter(x=out['q_disc'], y=out['iq_disc'], mode='markers', name='Discrete Peaks'),
            row=2, col=1
        )
        # Subplot 3: Combined view with both continuous and discrete data
        fig.add_trace(
            go.Scatter(x=out['q'], y=out['iq'], mode='lines', name='Continuous PXRD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=out['q_disc'], y=out['iq_disc'], mode='markers', name='Discrete Peaks'),
            row=3, col=1
        )
        fig.update_layout(
            title='Interactive PXRD Plot',
            xaxis_title='Q (Momentum Transfer)',
            yaxis_title='Intensity',
            height=900  # Adjust the height as needed for three subplots
        )
        return fig
    else:
        return None

# Spacegroup dropdown
with open("decifer/spacegroups.txt", "r") as f:
    space_groups = [line.strip() for line in f if line.strip()] # Remove empty lines
spacegroup_dropdown = [{"label": sg, "value": f"{sg}_sg"} for sg in space_groups]

# Default structure
structure_component = ctc.StructureMoleculeComponent(id="structure-viewer")

# Layout
layout = html.Div([
    # Header with a bigger title
    html.Header([
        html.H1([
            "deCIFer",
            html.Span("demo", style={
                "fontSize": "0.5em",  # Smaller font size for superscript effect
                "fontWeight": "lighter",
                "position": "relative",
                "top": "-1.0em",  # Moves the text upwards
                "marginLeft": "5px"  # Slight spacing from main text
            })
        ], style={
            "textAlign": "left",
            "margin": "0",
            "padding": "20px",
            "fontFamily": "Helvetica Neue, sans-serif",
            "fontSize": "3em",
            "fontWeight": "bold",
        })
    ], style={
        "backgroundColor": "#343a40",
        "color": "white",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "gap": "0px",
        "margin-bottom": "20px",
    }),

    # Top row: Three columns (manual input, CIF string, crystal visualization)
    html.Div([
        html.Div([
            html.H3("Structure Generation Input", style={"textAlign": "center", "fontFamily": 'Helvetica Neue, sans-serif', "fontSize": "20px", "margin-bottom": "10px"}),
            # Left column: Manual Structure Input Form
            html.Div([
                # Upload input
                html.Div([
                    du.Upload(
                        id="upload-cif",
                        text="Upload CIF",
                        max_files=1,
                        filetypes=["cif"],
                        default_style={
                            "width": "200px",
                            "backgroundColor": "black",
                            "color": "white",
                            "borderRadius": "5px",
                            "lineHeight": "30px",  # Matches height for centering
                            "textAlign": "center",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "16px",  # Adjusted for a smaller button
                            "fontWeight": "bold",
                            "padding": "2px 5px",  # Keeps it compact
                            "minHeight": "15px",
                        }
                    ),
                    du.Upload(
                        id="upload-pxrd",
                        text="Upload PXRD",
                        max_files=1,
                        filetypes=[".xy"],
                        default_style={
                            "width": "200px",
                            "backgroundColor": "black",
                            "color": "white",
                            "borderRadius": "5px",
                            "lineHeight": "30px",  # Matches height for centering
                            "textAlign": "center",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "16px",  # Adjusted for a smaller button
                            "fontWeight": "bold",
                            "padding": "2px 5px",  # Keeps it compact
                            "minHeight": "15px",
                        }
                    ),
                ], style={"marginBottom": "10px", "display": "flex", "justifyContent": "space-evenly"}),

                # Composition input
                html.Div([
                    html.Label("Composition (use 'X' for wildcards):"),
                    dcc.Input(id="composition-input", type="text", placeholder="e.g. A2BX6", style={"width": "100%"})
                ], style={"marginBottom": "10px"}),

                
                # Space group dropdown
                html.Div([
                    html.Label("Space Group:"),
                    dcc.Dropdown(
                        id="spacegroup-dropdown",
                        options=spacegroup_dropdown,
                        placeholder="Select a space group",
                        value="None", 
                    )
                ], style={"marginBottom": "10px"}),
                
                # Cell parameters inputs
                html.Div([
                    html.Label("Cell Parameters:"),
                    html.Div([
                        html.Div([
                            html.Label("a:"),
                            dcc.Input(id="cell-a", type="number", placeholder="a", style={"width": "90%"})
                        ], style={"display": "inline-block", "width": "30%", "paddingRight": "5px"}),
                        html.Div([
                            html.Label("b:"),
                            dcc.Input(id="cell-b", type="number", placeholder="b", style={"width": "90%"})
                        ], style={"display": "inline-block", "width": "30%", "paddingRight": "5px"}),
                        html.Div([
                            html.Label("c:"),
                            dcc.Input(id="cell-c", type="number", placeholder="c", style={"width": "90%"})
                        ], style={"display": "inline-block", "width": "30%"})
                    ], style={"display": "flex", "justifyContent": "space-between"}),
                    html.Div([
                        html.Div([
                            html.Label("α:"),
                            dcc.Input(id="cell-alpha", type="number", placeholder="alpha", style={"width": "90%"})
                        ], style={"display": "inline-block", "width": "30%", "paddingRight": "5px"}),
                        html.Div([
                            html.Label("β:"),
                            dcc.Input(id="cell-beta", type="number", placeholder="beta", style={"width": "90%"})
                        ], style={"display": "inline-block", "width": "30%", "paddingRight": "5px"}),
                        html.Div([
                            html.Label("γ:"),
                            dcc.Input(id="cell-gamma", type="number", placeholder="gamma", style={"width": "90%"})
                        ], style={"display": "inline-block", "width": "30%"})
                    ], style={"display": "flex", "justifyContent": "space-between", "marginTop": "10px"})
                ], style={"marginBottom": "10px"}),
                
                # Atoms input section
                html.Div([
                    html.Label("Atoms (Element, Multiplicity, x, y, z; Occupancy fixed at 1.0):"),
                    html.Div(
                        id="atoms-container",
                        children=[
                            html.Div([
                                dcc.Input(id={"type": "atom-element", "index": 0}, type="text", placeholder="Element", style={"width": "15%", "marginRight": "5px"}),
                                dcc.Input(id={"type": "atom-multiplicity", "index": 0}, type="number", placeholder="Multiplicity", style={"width": "15%", "marginRight": "5px"}),
                                dcc.Input(id={"type": "atom-x", "index": 0}, type="number", placeholder="x", style={"width": "15%", "marginRight": "5px"}),
                                dcc.Input(id={"type": "atom-y", "index": 0}, type="number", placeholder="y", style={"width": "15%", "marginRight": "5px"}),
                                dcc.Input(id={"type": "atom-z", "index": 0}, type="number", placeholder="z", style={"width": "15%", "marginRight": "5px"}),
                                dcc.Input(value="1.0", disabled=True, type="number", placeholder="Occupancy", style={"width": "15%"})
                            ], style={"marginBottom": "5px"})
                        ]
                    ),
                    html.Button("Add Atom", id="add-atom-button", n_clicks=0, style={"marginTop": "5px"})
                ], style={"marginBottom": "20px"}),
                # Generate button
                html.Div([
                        html.Button(
                            "🚀 Generate",  # Added an emoji for a more interactive feel
                            id="generate-button",
                            n_clicks=0,
                            style={
                                "padding": "10px 20px",
                                "fontSize": "16px",
                                "borderRadius": "8px",
                                "backgroundColor": "#007bff",
                                "color": "white",
                                "border": "none",
                                "cursor": "pointer",
                                "transition": "0.3s",
                            }
                        ),
                        dcc.Loading(
                            id="loading",
                            type="dot",
                            children=[html.Div(id="loading-div", style={"display": "none"})],
                        ),
                    ], style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "gap": "50px",  # Space between button and loading indicator
                        "marginTop": "10px"
                    }),
                # Error div
                html.Div(id="error-div"),

            ], style={
                "width": "525px",
                "padding": "10px",
                "boxShadow": "0px 0px 5px #ccc",
                "borderRadius": "10px",
                "backgroundColor": "#f8f9fa",
                "fontFamily": "Courier New, monospace",
                "overflowY": "auto",
                "minHeight": "800px",
                "maxHeight": "800px"
            }),
        ]),
            
        html.Div([
            html.H3("CIF Display", style={"textAlign": "center", "fontFamily": 'Helvetica Neue, sans-serif', "fontSize": "20px", "margin-bottom": "10px"}),
            
            # Middle column: CIF string display
            html.Div(
                id="cif-string-container",
                style={
                    "width": "525px",
                    "padding": "10px",
                    "boxShadow": "0px 0px 5px #ccc",
                    "borderRadius": "10px",
                    "backgroundColor": "#f8f9fa",
                    "fontFamily": "Courier New, monospace",
                    "overflowY": "auto",
                    "minHeight": "800px",
                    "maxHeight": "800px",
                }
            ),
        ]),
        
        html.Div([
            html.H3("Crystal Display", style={"textAlign": "center", "fontFamily": 'Helvetica Neue, sans-serif', "fontSize": "20px", "margin-bottom": "10px"}),
            
            html.Div([
                # Right column: Crystal Visualization
                html.Div([
                    structure_component.layout(),
                ],
                    id="crystal-vis-container",
                    style={
                        "width": "525px",
                        "padding": "10px",
                        "boxShadow": "0px 0px 5px #ccc",
                        "borderRadius": "10px",
                        "backgroundColor": "white",
                        "minHeight": "525px",
                        "maxHeight": "525px"
                    }
                ),
                html.Div([
                    dcc.Graph(
                       id="crystal-vis-container-pxrd-plot",
                       config = {
                           "staticPlot": False,
                           "displayModeBar": False,
                       }
                    )
                ],
                    id="crystal-vis-container-pxrd",
                    style={
                        "width": "525px",
                        "padding": "10px",
                        "boxShadow": "0px 0px 5px #ccc",
                        "borderRadius": "10px",
                        "backgroundColor": "white",
                        "minHeight": "260px",
                        "maxHeight": "260px"
                    }
                ),
            ], style = {"display": "flex",
                        "flex-direction": "column",
                        "justifyContent": "space-between",
                        "gap": "10px",
            }),
        ]),
    ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "flex-start",
            "maxWidth": "1600px",
            "margin": "1px auto"
    }),
])

def custom_generate_function(number, cif_string):
    """
    Dummy custom function to simulate processing.
    Replace this with your actual processing.
    """
    updated_cif = cif_string + f"\n\nGenerated with number: {number}"
    updated_vis = html.Div(
        f"Crystal visualization updated with number {number}",
        style={"textAlign": "center", "fontWeight": "bold"}
    )
    return updated_cif, updated_vis

from dash.dependencies import Input, Output, State
from dash import ctx
import dash

def create_structure_display(cif_string):

    cif_display = html.Pre(cif_string, style={
        'whiteSpace': 'pre-wrap',
        'maxHeight': '775px',
        'overflow': 'auto',
        'backgroundColor': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd'
    })

    # Make structure from cif_string
    structure = Structure.from_str(cif_string, fmt='cif')

    return cif_display, structure

import torch
from warnings import warn

# Function to load model from a checkpoint
def load_model_from_checkpoint(ckpt_path, device):
    
    # Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)  # Load checkpoint
    state_dict = checkpoint.get("best_model_state", checkpoint.get("best_model"))
    
    model_args = checkpoint["model_args"]

    # Map renamed keys
    renamed_keys = {
        'cond_size': 'condition_size',
        'condition_with_mlp_emb': 'condition',
    }
    for old_key, new_key in renamed_keys.items():
        if old_key in model_args:
            model_args['use_old_model_format'] = True
            warn(
                f"'{old_key}' is deprecated and has been renamed to '{new_key}'. "
                "Please update your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2
            )
            model_args[new_key] = model_args.pop(old_key)
    
    # Remove unused keys
    removed_keys = [
        'use_lora',
        'lora_rank',
        'condition_with_cl_emb',
        'cl_model_ckpt',
        'freeze_condition_embedding',
    ]
    for removed_key in removed_keys:
        if removed_key in model_args:
            warn(
                f"'{removed_key}' is no longer used and will be ignored. "
                "Consider removing it from your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2
            )
            model_args.pop(removed_key)

    
    # Load the model and checkpoint
    model_config = DeciferConfig(**model_args)
    model = Decifer(model_config).to(device)
    model.device = device
    
    # Fix the keys of the state dict per CrystaLLM
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    #model.load_state_dict(state_dict)  # Load modified state_dict into the model
    model.load_state_dict(state_dict)
    return model

from bin.train import TrainConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    reinstate_symmetry_loop, 
    extract_space_group_symbol, 
    replace_symmetry_loop_with_P1, 
)

# Tokenizer, get start, padding and newline IDs
TOKENIZER = Tokenizer()
VOCAB_SIZE = TOKENIZER.vocab_size
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]
SPACEGROUP_ID = TOKENIZER.token_to_id["_symmetry_space_group_name_H-M"]
DECODE = TOKENIZER.decode
ENCODE = TOKENIZER.encode

def extract_prompt(sequence, device, add_composition=True, add_spacegroup=False):

    # Find "data_" and slice
    try:
        end_prompt_index = np.argwhere(sequence == START_ID)[0][0] + 1
    except IndexError:
        raise ValueError(f"'data_' id: {START_ID} not found in sequence", DECODE(sequence))

    # Add composition (and spacegroup)
    if add_composition:
        end_prompt_index += np.argwhere(sequence[end_prompt_index:] == NEWLINE_ID)[0][0]

        if add_spacegroup:
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == SPACEGROUP_ID)[0][0]
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == NEWLINE_ID)[0][0]
            
        end_prompt_index += 1
    
    prompt_ids = torch.tensor(sequence[:end_prompt_index].long()).to(device=device)

    return prompt_ids

def rwp_fn(sample, gen):
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))

@app.callback(
    Output("cif-string-container", "children"),
    Output(structure_component.id(), "data"),
    Output("crystal-vis-container-pxrd-plot", "figure"),
    Output("crystal-vis-container-pxrd-plot", "style"),
    Output("loading-div", "children"),
    Output("error-div", "children"),
    Input("upload-cif", "isCompleted"),
    Input("upload-cif", "fileNames"),
    Input("upload-pxrd", "isCompleted"),
    Input("upload-pxrd", "fileNames"),
    Input("generate-button", "n_clicks"),
    State("composition-input", "value"),
    State("spacegroup-dropdown", "value"),
    State("cell-a", "value"),
    State("cell-b", "value"),
    State("cell-c", "value"),
    State("cell-alpha", "value"),
    State("cell-beta", "value"),
    State("cell-gamma", "value"),
    State({"type": "atom-element", "index": dash.ALL}, "value"),
    State({"type": "atom-multiplicity", "index": dash.ALL}, "value"),
    State({"type": "atom-x", "index": dash.ALL}, "value"),
    State({"type": "atom-y", "index": dash.ALL}, "value"),
    State({"type": "atom-z", "index": dash.ALL}, "value"),
)
def generate_structures(
    cif_completed,
    cif_names,
    pxrd_completed,
    pxrd_names,
    #model_completed,
    #model_names,
    generate_button_clicks,
    composition_string,
    spacegroup_string,
    cell_a_value,
    cell_b_value,
    cell_c_value,
    cell_alpha_value,
    cell_beta_value,
    cell_gamma_value,
    atoms_element,
    atoms_mult,
    atoms_x,
    atoms_y,
    atoms_z,
):
    # Generation button triggered:
    if ctx.triggered_id == "generate-button": #model_completed and model_names:

        #TODO: make same system for PXRD input, involves downsampling to correct q-grid
        
        fig = make_subplots(rows=1, cols=1)
    
        # Look for CIF:
        if cif_completed and cif_names:

            cif_paths = glob(os.path.join(UPLOAD_FOLDER, "*", cif_names[0]))
            if cif_paths:
                cif_paths.sort(key=os.path.getmtime, reverse=True)
                cif_path = cif_paths[0]
            else:
                raise dash.exceptions.PreventUpdate

            # Create PXRD input from input CIF 
            structure = Structure.from_file(cif_path)
            cif_string = CifWriter(struct=structure, symprec=0.1).__str__()
            pxrd = generate_continuous_xrd_from_cif(cif_string, qmin=0.0, qmax=10.0, debug=True)
            if pxrd is not None:
                cond_vec = torch.from_numpy(pxrd['iq']).unsqueeze(0).to(device='cuda')
            else:
                cond_vec = None
            pxrd_ref = generate_continuous_xrd_from_cif(cif_string, qmin=0.5, qmax=7.5, debug=True)
            
            # Display PXRD
            if pxrd_ref is not None:
                fig.add_traces([
                    go.Scatter(x=pxrd_ref['q'], y=pxrd_ref['iq'], mode='lines', name='Reference'),
                ])
                fig.update_layout(
                    xaxis_title='Q [Å^-1]',
                    yaxis_title='I(Q) [a.u.]',
                    height=225,
                    width=475,
                    yaxis = dict(tickvals=[]),
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor = 'rgba(0, 0, 0, 0)',
                    paper_bgcolor = 'rgba(0, 0, 0, 0)',
                    legend=dict(x=0.8, y=1.0),
                )
            else:
                raise Exception("Cannot read CIF")

        elif pxrd_completed and pxrd_names:

            # Do something
            cond_vec = None
            pxrd_ref = None
        else:
            # Do nothing
            cond_vec = None
            pxrd_ref = None
            
        # Load model
        #if model_completed and model_names:
        #    model_paths = glob(os.path.join(UPLOAD_FOLDER, "*", model_names[0]))
        #    if model_paths:
        #        model_paths.sort(key=os.path.getmtime, reverse=True)
        if True: # NOTE: TEMP
            if cond_vec is None:
                model_paths = ["../../phd_projects/deCIFer/experiments/model__nocond__context_3076__robust_full_trainingcurves/ckpt.pt"]
            else:
                model_paths = ["../../phd_projects/deCIFer/experiments/model__conditioned_mlp_augmentation__context_3076__robust_full_trainingcurves/ckpt.pt"]
            model = load_model_from_checkpoint(model_paths[0], device='cuda')

            # Make atoms
            atoms = []
            for i, (el, mtpl, x, y, z) in enumerate(zip(atoms_element, atoms_mult, atoms_x, atoms_y, atoms_z)):
                if not None in [el, mtpl, x, y, z]:
                    atoms.append(f"{el} {el}{i} {int(mtpl)} {x:.4f} {y:.4f} {z:.4f} 1.0000")

            # Generate custom CIF
            generated = model.generate_custom(
                idx = torch.tensor([START_ID]).unsqueeze(0).to(device='cuda'),
                max_new_tokens=3076,
                cond_vec = cond_vec,
                start_indices_batch=[[0]],
                composition_string = composition_string,
                spacegroup_string = spacegroup_string,
                cell_a_string = f'{cell_a_value:.4f}' if cell_a_value is not None else None,
                cell_b_string = f'{cell_b_value:.4f}' if cell_b_value is not None else None,
                cell_c_string = f'{cell_c_value:.4f}' if cell_c_value is not None else None,
                cell_alpha_string = f'{cell_alpha_value:.4f}' if cell_alpha_value is not None else None,
                cell_beta_string = f'{cell_beta_value:.4f}' if cell_beta_value is not None else None,
                cell_gamma_string = f'{cell_gamma_value:.4f}' if cell_gamma_value is not None else None,
                atoms_string_list = atoms if len(atoms) > 0 else None,
            ).cpu().numpy()

            generated = [ids[ids != PADDING_ID] for ids in generated]
                
            # Convert to string
            cif_string_gen = DECODE(generated[0])
            cif_string_gen = replace_symmetry_loop_with_P1(cif_string_gen)
            spacegroup_symbol = extract_space_group_symbol(cif_string_gen)
            if spacegroup_symbol != "P 1":
                cif_string_gen = reinstate_symmetry_loop(cif_string_gen, spacegroup_symbol)

            # Calculate PXRD
            pxrd_gen = generate_continuous_xrd_from_cif(cif_string_gen, qmin=0.5, qmax=7.5, debug=True, fwhm_range=(0.05, 0.05))

            if pxrd_gen is not None:
                fig.add_traces([
                    go.Scatter(x=pxrd_gen['q'], y=pxrd_gen['iq'], mode='lines', name='Generated'),
                ])
                if pxrd_ref is not None:
                    fig.add_traces([
                        go.Scatter(x=pxrd_gen['q'], y=pxrd_ref['iq'] - pxrd_gen['iq'] - 0.5, mode='lines', name=f'Difference, Rwp:{rwp_fn(pxrd_ref['iq'], pxrd_gen['iq']):1.3f}'),
                    ])

                fig.update_layout(
                    xaxis_title='Q [Å^-1]',
                    yaxis_title='I(Q) [a.u.]',
                    height=225,
                    width=475,
                    yaxis = dict(tickvals=[]),
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor = 'rgba(0, 0, 0, 0)',
                    paper_bgcolor = 'rgba(0, 0, 0, 0)',
                    legend=dict(x=0.8, y=1.0),
                )

            # Get display, TODO: Make it show the best structure
            cif_display, new_structure = create_structure_display(cif_string_gen)

            return (
                cif_display, 
                new_structure,
                fig,
                {},
                "",
                html.Div(),
            )
        else:
            return (
                html.Div(),
                None,
                fig,
                {},
                "",
                html.Div(),
            )
    else:
        return (
            html.Div(),
            None,
            go.Figure(),
            {"display": "none"},
            "",
            html.Div(),
        )

@app.callback(
    Output("atoms-container", "children"),
    Input("add-atom-button", "n_clicks"),
    State("atoms-container", "children")
)
def add_atom_row(n_clicks, children):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    new_index = len(children)
    new_atom = html.Div([
        dcc.Input(id={"type": "atom-element", "index": new_index}, type="text", placeholder="Element", style={"width": "15%", "marginRight": "5px"}),
        dcc.Input(id={"type": "atom-multiplicity", "index": new_index}, type="number", placeholder="Multiplicity", style={"width": "15%", "marginRight": "5px"}),
        dcc.Input(id={"type": "atom-x", "index": new_index}, type="number", placeholder="x", style={"width": "15%", "marginRight": "5px"}),
        dcc.Input(id={"type": "atom-y", "index": new_index}, type="number", placeholder="y", style={"width": "15%", "marginRight": "5px"}),
        dcc.Input(id={"type": "atom-z", "index": new_index}, type="number", placeholder="z", style={"width": "15%", "marginRight": "5px"}),
        dcc.Input(value="1.0", disabled=True, type="number", placeholder="Occupancy", style={"width": "15%"})
    ], style={"marginBottom": "5px"})
    children.append(new_atom)
    return children


# Register
ctc.register_crystal_toolkit(app, layout=layout)

if __name__ == "__main__":
    app.run_server(debug=True)
