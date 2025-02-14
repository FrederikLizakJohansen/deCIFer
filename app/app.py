import base64
import os

import dash
from dash import Dash
from dash import Input, Output, State, ctx, html, Dash, dcc
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import torch
import numpy as np

from pymatgen.core.structure import Structure

# Import from local modules
from bin.evaluate import load_model_from_checkpoint
from bin.train import TrainConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    generate_continuous_xrd_from_cif,
    replace_symmetry_loop_with_P1,
    extract_space_group_symbol,
    reinstate_symmetry_loop
)

import dash_uploader as du
import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

# Instantiate the app and override index_string
app = Dash(__name__, assets_folder=SETTINGS.ASSETS_PATH)
server = app.server

# Configure Dash Uploader
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
du.configure_upload(app, UPLOAD_FOLDER)

PADDING_ID = Tokenizer().padding_id
START_ID = Tokenizer().token_to_id["data_"]
DECODE = Tokenizer().decode

def rwp_fn(sample, gen):
    """
    Example function to compute Rwp-like difference metric between arrays.
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))

# Create the structure component once so it can be referenced in callbacks
structure_component = ctc.StructureMoleculeComponent(id="structure-viewer")
STRUCTURE_COMPONENT_ID = structure_component.id()

FWHM = 0.05

CS = [ 
    "#A69580", # Background
    "#0D0D0D", # Top bar
    "#F2F2F2", 
    "#F2C166", # Component smaller sections
    "#F2D06B", # Component background
]
CS = [
    "#F0F5F5",  # Background (soft pastel green)
    "#2E2E2E",  # Top bar (deep charcoal for contrast)
    "black",  # Component smaller sections (pastel peach)
    "#F5F0F4",  # Component background (muted sand)
    "white",  # Light accent (gentle off-white)
]
# Read in available space groups
with open("../decifer/spacegroups.txt", "r") as f:
    space_groups = [line.strip() for line in f if line.strip()]

spacegroup_dropdown = [{"label": sg, "value": f"{sg}_sg"} for sg in space_groups]

structure_component = ctc.StructureMoleculeComponent(id="structure-viewer")

# Define the periodic table layout (rows of 18 cells; use None for blank cells)
periodic_table_layout = [
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
    # Break
    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    # Lanthanides (Period 6)
    [None, None, None, "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", None],
    # Actinides (Period 7)
    [None, None, None, "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", None]
]

# Build a list of all unique element symbols (default: all are selected)
inactive_elements_dict = {}
for row in periodic_table_layout:
    for el in row:
        if el is not None:
            inactive_elements_dict[el] = False
            # inactive_elements_dict['element'] = el
            # inactive_elements_dict['inactive'] = False

# Define element group mappings (for the elements present in the table)
element_to_group = {
    "H": "nonmetal",
    "He": "noble",
    "Li": "alkali",
    "Be": "alkaline",
    "B": "metalloid",
    "C": "nonmetal",
    "N": "nonmetal",
    "O": "nonmetal",
    "F": "halogen",
    "Ne": "noble",
    "Na": "alkali",
    "Mg": "alkaline",
    "Al": "post-transition",
    "Si": "metalloid",
    "P": "nonmetal",
    "S": "nonmetal",
    "Cl": "halogen",
    "Ar": "noble",
    "K": "alkali",
    "Ca": "alkaline",
    "Sc": "transition",
    "Ti": "transition",
    "V": "transition",
    "Cr": "transition",
    "Mn": "transition",
    "Fe": "transition",
    "Co": "transition",
    "Ni": "transition",
    "Cu": "transition",
    "Zn": "transition",
    "Ga": "post-transition",
    "Ge": "metalloid",
    "As": "metalloid",
    "Se": "nonmetal",
    "Br": "halogen",
    "Kr": "noble",
    "Rb": "alkali",
    "Sr": "alkaline",
    "Y": "transition",
    "Zr": "transition",
    "Nb": "transition",
    "Mo": "transition",
    "Tc": "transition",
    "Ru": "transition",
    "Rh": "transition",
    "Pd": "transition",
    "Ag": "transition",
    "Cd": "transition",
    "In": "post-transition",
    "Sn": "post-transition",
    "Sb": "metalloid",
    "Te": "metalloid",
    "I": "halogen",
    "Xe": "noble",
    "Cs": "alkali",
    "Ba": "alkaline",
    "La": "lanthanide",
    "Hf": "transition",
    "Ta": "transition",
    "W": "transition",
    "Re": "transition",
    "Os": "transition",
    "Ir": "transition",
    "Pt": "transition",
    "Au": "transition",
    "Hg": "transition",
    "Tl": "post-transition",
    "Pb": "post-transition",
    "Bi": "post-transition",
    "Po": "metalloid",
    "At": "halogen",
    "Rn": "noble",
    "Fr": "alkali",
    "Ra": "alkaline",
    "Rf": "transition",
    "Db": "transition",
    "Sg": "transition",
    "Bh": "transition",
    "Hs": "transition",
    "Mt": "transition",
    "Ds": "transition",
    "Rg": "transition",
    "Cn": "transition",
    "Nh": "post-transition",
    "Fl": "post-transition",
    "Mc": "post-transition",
    "Lv": "post-transition",
    "Ts": "halogen",
    "Og": "noble",
    "Ce": "lanthanide",
    "Pr": "lanthanide",
    "Nd": "lanthanide",
    "Pm": "lanthanide",
    "Sm": "lanthanide",
    "Eu": "lanthanide",
    "Gd": "lanthanide",
    "Tb": "lanthanide",
    "Dy": "lanthanide",
    "Ho": "lanthanide",
    "Er": "lanthanide",
    "Tm": "lanthanide",
    "Yb": "lanthanide",
    "Lu": "lanthanide",
    "Ac": "actinide",
    "Th": "actinide",
    "Pa": "actinide",
    "U": "actinide",
    "Np": "actinide",
    "Pu": "actinide",
    "Am": "actinide",
    "Cm": "actinide",
    "Bk": "actinide",
    "Cf": "actinide",
    "Es": "actinide",
    "Fm": "actinide",
    "Md": "actinide",
    "No": "actinide",
    "Lr": "actinide"
}

group_colors = {
    "alkali": "#FF6666",         # red-ish
    "alkaline": "#FFB347",       # orange-ish
    "transition": "#B0C4DE",     # light steel blue
    "post-transition": "#CCCCCC",# grey
    "metalloid": "#CCCC99",      # olive-ish
    "nonmetal": "#90EE90",       # light green
    "halogen": "#66CDAA",        # medium aquamarine
    "noble": "#87CEFA",          # light sky blue
    "lanthanide": "#FFB6C1",     # light pink
    "actinide": "#FFA07A",       # light salmon
}

PERIODIC_DIM = "30px"
PERIODIC_FONTSIZE = "1.0em"

def create_periodic_table():
    """Return an HTML table representing the periodic table with clickable element cells colored by group."""
    table_rows = []
    # Reduce cell size to make the table less wide.
    cell_width = PERIODIC_DIM
    cell_height = PERIODIC_DIM
    for row in periodic_table_layout:
        # Check if the row is a break row (all cells are None)
        if all(cell is None for cell in row):
            table_rows.append(html.Tr(
                html.Td("", colSpan=len(row), style={"height": "10px", "border": "none"})
            ))
            continue

        cells = []
        for cell in row:
            if cell is None:
                cells.append(html.Td(""))  # Empty cell for spacing
            else:
                group = element_to_group.get(cell, "nonmetal")
                base_color = group_colors.get(group, "#4CAF50")
                cells.append(html.Td(
                    html.Button(
                        cell,
                        id={"type": "ptable-cell", "element": cell},
                        n_clicks=0,
                        style={
                            "width": cell_width,
                            "height": cell_height,
                            "margin": "0px",
                            "border": "0px solid #ccc",
                            "borderRadius": "0px",
                            "backgroundColor": base_color,
                            "color": "black",
                            "cursor": "pointer",
                            "fontSize": PERIODIC_FONTSIZE,
                            "textAlign": "center",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                        }
                    )
                ))
        table_rows.append(html.Tr(cells))
    return html.Table(table_rows, style={"borderCollapse": "collapse", "width": "auto", "marginBottom": "0px"})

layout = html.Div([
    html.Header([
        html.H1([
            "deCIFer",
            html.Span("demo", style={
                "fontSize": "0.5em",
                "fontWeight": "lighter",
                "position": "relative",
                "top": "-1.0em",
                "marginLeft": "5px"
            })
        ], style={
            "textAlign": "left",
            "margin": "0",
            "padding": "10px",
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
        "margin-bottom": "10px",
        "background-color": CS[1],
    }),

    # Top row with three columns
    html.Div([
        # Left column: input forms
        html.Div([
            html.H1("Structure Generation Input", style={
                "textAlign": "center", 
                "fontFamily": 'Helvetica Neue, sans-serif', 
                "fontSize": "30px", 
                "fontWeight": "bold",
                "margin-bottom": "5px",
                "color": CS[2]
            }),

            html.Div([

                html.Div([
                    html.H2("Composition:", style={
                        #"fontFamily": 'Verdana, sans-serif', 
                        "fontSize": "16px", 
                        "margin-bottom": "0px"
                    }),
                    dcc.Input(id="composition-input", type="text", placeholder="Enter a composition (e.g. CdCl2)", style={"width": "100%", "height": "30px", "fontSize": "14px", "borderRadius": "5px", "padding": "5px"})
                ], style={"marginBottom": "10px", "display": "flex", "justifyContent": "center", "gap": "10px", "padding": "10px", "background-color": CS[3], "alignItems": "center", "borderRadius": "5px"}),#, "flexDirection": "column", "alignItems": "center"}),

                html.Div([
                    html.Label("Space Group:",style={"white-space": "nowrap"}),
                    dcc.Dropdown(
                        id="spacegroup-dropdown",
                        options=spacegroup_dropdown,
                        placeholder="Select a space group",
                        value="None",
                        style={"width": "100%"}
                    )
                ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "10px", "padding": "10px", "background-color": CS[3]}),

                # Cell parameters
                html.Div([
                    html.Label("Cell Parameters:"),
                    html.Div([
                        html.Div(
                            id = "cell-a-display",
                            children = [
                            html.Label("a:"),
                            dcc.Input(id="cell-a", type="text", placeholder="Enter a number", step="any", style={"width": "90%", "fontSize": "14px", "height": "30px", "borderRadius": "5px", "padding": "5px"}),
                            html.Button(
                                "x",
                                id="clear-btn-a",
                                n_clicks=0,
                                tabIndex="-1",
                                style={
                                    "height": "20px",
                                    "width": "20px",
                                    "fontSize": "20px",
                                    "background-color": "black",
                                    "color": "white",
                                    "borderRadius": "50%",
                                    "display": "inline-flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "textAlign": "center",
                                    "paddingBottom": "8px",
                                    "userSelect": "none",
                                    "cursor": "pointer",
                                }
                            ),
                        ], style={"display": "flex", "width": "100%", "paddingRight": "5px", "alignItems": "center", "gap": "5px"}),
                        html.Div([
                            html.Label("b:"),
                            dcc.Input(id="cell-b", type="text", placeholder="Enter a number", step="any", style={"width": "90%", "fontSize": "14px", "height": "30px", "borderRadius": "5px", "padding": "5px"}),
                            html.Button(
                                "x",
                                id="clear-btn-b",
                                n_clicks=0,
                                tabIndex="-1",
                                style={
                                    "height": "20px",
                                    "width": "20px",
                                    "fontSize": "20px",
                                    "background-color": "black",
                                    "color": "white",
                                    "borderRadius": "50%",
                                    "display": "inline-flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "textAlign": "center",
                                    "paddingBottom": "8px",
                                    "userSelect": "none",
                                    "cursor": "pointer",
                                }
                            ),
                        ], style={"display": "flex", "width": "100%", "paddingRight": "5px", "alignItems": "center", "gap": "5px"}),
                        html.Div([
                            html.Label("c:"),
                            dcc.Input(id="cell-c", type="text", placeholder="Enter a number", step="any", style={"width": "90%", "fontSize": "14px", "height": "30px", "borderRadius": "5px", "padding": "5px"}),
                            html.Button(
                                "x",
                                id="clear-btn-c",
                                n_clicks=0,
                                tabIndex="-1",
                                style={
                                    "height": "20px",
                                    "width": "20px",
                                    "fontSize": "20px",
                                    "background-color": "black",
                                    "color": "white",
                                    "borderRadius": "50%",
                                    "display": "inline-flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "textAlign": "center",
                                    "paddingBottom": "8px",
                                    "userSelect": "none",
                                    "cursor": "pointer",
                                }
                            ),
                        ], style={"display": "flex", "width": "100%", "paddingRight": "5px", "alignItems": "center", "gap": "5px"}),
                    ], style={"display": "flex", "gap": "10px"}),
                    html.Div([
                        html.Div([
                            html.Label("α:"),
                            dcc.Input(id="cell-alpha", type="text", placeholder="Enter a number", step="any", style={"width": "90%", "fontSize": "14px", "height": "30px", "borderRadius": "5px", "padding": "5px"}),
                            html.Button(
                                "x",
                                id="clear-btn-alpha",
                                n_clicks=0,
                                tabIndex="-1",
                                style={
                                    "height": "20px",
                                    "width": "20px",
                                    "fontSize": "20px",
                                    "background-color": "black",
                                    "color": "white",
                                    "borderRadius": "50%",
                                    "display": "inline-flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "textAlign": "center",
                                    "paddingBottom": "8px",
                                    "userSelect": "none",
                                    "cursor": "pointer",
                                }
                            ),
                        ], style={"display": "flex", "width": "100%", "paddingRight": "5px", "alignItems": "center", "gap": "5px"}),
                        html.Div([
                            html.Label("β:"),
                            dcc.Input(id="cell-beta", type="text", placeholder="Enter a number", step="any", style={"width": "90%", "fontSize": "14px", "height": "30px", "borderRadius": "5px", "padding": "5px"}),
                            html.Button(
                                "x",
                                id="clear-btn-beta",
                                n_clicks=0,
                                tabIndex="-1",
                                style={
                                    "height": "20px",
                                    "width": "20px",
                                    "fontSize": "20px",
                                    "background-color": "black",
                                    "color": "white",
                                    "borderRadius": "50%",
                                    "display": "inline-flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "textAlign": "center",
                                    "paddingBottom": "8px",
                                    "userSelect": "none",
                                    "cursor": "pointer",
                                }
                            ),
                        ], style={"display": "flex", "width": "100%", "paddingRight": "5px", "alignItems": "center", "gap": "5px"}),
                        html.Div([
                            html.Label("γ:"),
                            dcc.Input(id="cell-gamma", type="text", placeholder="Enter a number", step="any", style={"width": "90%", "fontSize": "14px", "height": "30px", "borderRadius": "5px", "padding": "5px"}),
                            html.Button(
                                "x",
                                id="clear-btn-gamma",
                                n_clicks=0,
                                tabIndex="-1",
                                style={
                                    "height": "20px",
                                    "width": "20px",
                                    "fontSize": "20px",
                                    "background-color": "black",
                                    "color": "white",
                                    "borderRadius": "50%",
                                    "display": "inline-flex",
                                    "flexDirection": "column",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "textAlign": "center",
                                    "paddingBottom": "8px",
                                    "userSelect": "none",
                                    "cursor": "pointer",
                                }
                            ),
                        ], style={"display": "flex", "width": "100%", "paddingRight": "5px", "alignItems": "center", "gap": "5px"}),
                    ], style={"display": "flex", "marginTop": "10px", "gap": "10px"})
                ], style={"marginBottom": "10px", "padding": "10px", "background-color": CS[3]}),

                # Atoms input
                html.Div([
                    html.Label("Atoms (Element, Multiplicity, x, y, z; Occupancy=1.0):"),
                    html.Div(
                        id="atoms-container",
                        children=[
                            html.Div([
                                dcc.Input(
                                    id={"type": "atom-element", "index": 0},
                                    type="text",
                                    placeholder="Element",
                                    style={"width": "15%", "marginRight": "5px"}
                                ),
                                dcc.Input(
                                    id={"type": "atom-multiplicity", "index": 0},
                                    type="number",
                                    placeholder="Multiplicity",
                                    style={"width": "15%", "marginRight": "5px"}
                                ),
                                dcc.Input(
                                    id={"type": "atom-x", "index": 0},
                                    type="number",
                                    placeholder="x",
                                    style={"width": "15%", "marginRight": "5px"}
                                ),
                                dcc.Input(
                                    id={"type": "atom-y", "index": 0},
                                    type="number",
                                    placeholder="y",
                                    style={"width": "15%", "marginRight": "5px"}
                                ),
                                dcc.Input(
                                    id={"type": "atom-z", "index": 0},
                                    type="number",
                                    placeholder="z",
                                    style={"width": "15%", "marginRight": "5px"}
                                ),
                                dcc.Input(
                                    value="1.0",
                                    disabled=True,
                                    type="number",
                                    placeholder="Occupancy",
                                    style={"width": "5%"}
                                )
                            ], style={"marginBottom": "5px"})
                        ]
                    ),
                    html.Button("Add Atom", id="add-atom-button", n_clicks=0, style={"marginTop": "5px"})
                ], style={"marginBottom": "10px", "background-color": CS[3], "padding": "10px"}),
                
                # ---- New Periodic Table Section Start ----
                dcc.Store(id="inactive-elements-store", data=inactive_elements_dict),
                dcc.Store(id="element-to-group-store", data=element_to_group),
                dcc.Store(id="group-colors-store", data=group_colors),
                html.Div([
                    html.Div([
                        html.H3("Select elements from periodic table:", style={"textAlign": "left", "marginBottom": "10px"}),
                        html.Div([
                        html.Button("Select All", id="select-all", n_clicks=0, style={"marginRight": "5px"}),
                        html.Button("Unselect All", id="unselect-all", n_clicks=0)
                        ]),
                    ], style={"marginTop": "10px", "textAlign": "center", "marginBottom": "10px", "display": "flex", "justifyContent": "space-between", "marginLeft": "5px", "marginRight": "10px"}),
                    
                    html.Div(
                    [create_periodic_table()],
                        style={"display": "flex", "alignItems": "center", "justifyContent": "center"},
                    ),
                ], style={"background-color": CS[3], "padding": "10px", "marginBottom": "10px"}),
                
                html.Div([
                    html.Div([
                        # Upload button for CIF
                        dcc.Upload(
                            id="upload-cif",
                            children=html.Button(
                                "⬆️ Upload CIF",
                                style={"fontSize": "16px", "padding": "12px 24px"}  # Adjust size
                            ),
                            multiple=False,
                            accept=".cif"
                        ),
                            html.Div(id="file-name-display-cif", style={"marginTop": "10px", "fontSize": "14px", "alignText": "center"}),

                    ], style = {"display": "flex", "flexDirection": "column", "marginRight": "10px", "alignItems": "center"}),
                    # Upload button for PXRD

                    html.Div([
                        dcc.Upload(
                            id="upload-pxrd",
                            children=html.Button(
                                "⬆️ Upload PXRD",
                                style={"fontSize": "16px", "padding": "12px 24px"}  # Adjust size
                            ),
                            multiple=False,
                            accept=".xy"
                        ),
                            html.Div(id="file-name-display-pxrd", style={"marginTop": "10px", "fontSize": "14px", "alignText": "center"}),

                    ], style = {"display": "flex", "flexDirection": "column", "marginRight": "10px", "alignItems": "center"}),

                ], style={"marginBottom": "10px", "display": "flex", "justifyContent": "space-evenly"}),
                html.Div([
                    html.Div([
                        # Generate button
                        html.Button(
                            id="generate-button",
                            children = [html.Span("🚀 Generate")],
                            style={"fontSize": "25px", "padding": "12px 24px", "borderRadius": "5px", "cursor": "pointer"},  # Adjust size
                            n_clicks=0,
                        ),
                        dcc.Loading(
                            id="loading",
                            type="dot",
                            color="black",  # Set loading dot color to black
                            children=[html.Div(id="loading-div", style={"display": "none"})],
                            style={
                                "position": "absolute",
                                "top": "50%",
                                "left": "50%",
                                "transform": "translate(-50%, -70%)",
                                "zIndex": 2
                            }
                        ),
                        ], style = {"display": "flex", "flexDirection": "column", "marginRight": "10px", "alignItems": "center"}),

                ], style={"marginBottom": "10px", "display": "flex", "justifyContent": "space-evenly"}),

                html.Div(id="error-div"),
            ], style={
                "width": "575px",
                "padding": "10px",
                "boxShadow": "0px 0px 5px #ccc",
                "borderRadius": "10px",
                "backgroundColor": CS[4],
                "fontFamily": "Courier New, monospace",
                "overflowY": "auto",
                "minHeight": "975px",
                "maxHeight": "975px",
            }),
        ]),

        # Middle column: CIF display
        html.Div([
            html.H3("CIF Display", style={
                "textAlign": "center", 
                "fontFamily": 'Helvetica Neue, sans-serif', 
                "fontSize": "30px", 
                "fontWeight": "bold",
                "margin-bottom": "5px",
                "color": CS[2],
            }),
            html.Div(
                id="cif-string-container",
                style={
                    "width": "425px",
                    "padding": "10px",
                    "boxShadow": "0px 0px 5px #ccc",
                    "borderRadius": "10px",
                    "backgroundColor": CS[4],
                    "fontFamily": "Courier New, monospace",
                    "overflowY": "auto",
                    "minHeight": "975px",
                    "maxHeight": "975px",
                }
            ),
        ]),

        # Right column: crystal + PXRD
        html.Div([
            html.H3("Crystal Display", style={
                "textAlign": "center", 
                "fontFamily": 'Helvetica Neue, sans-serif', 
                "fontSize": "30px", 
                "fontWeight": "bold",
                "margin-bottom": "5px",
                "color": CS[2]
            }),
            html.Div([
                html.Div([
                    structure_component.layout(),
                ],
                    id="crystal-vis-container",
                    style={
                        "width": "525px",
                        "padding": "10px",
                        "boxShadow": "0px 0px 5px #ccc",
                        "borderRadius": "10px",
                        "backgroundColor": CS[4],
                        "minHeight": "525px",
                        "maxHeight": "525px"
                    }
                ),
                html.Div([
                    dcc.Graph(
                       id="crystal-vis-container-pxrd-plot",
                       config={
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
                        "backgroundColor": CS[4],
                        "minHeight": "435px",
                        "maxHeight": "435px"
                    }
                ),
            ], style={
                "display": "flex",
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
], style={"background-color": CS[0]})

@app.callback(
    Output("cell-a", "value"),
    Output("cell-b", "value"),
    Output("cell-c", "value"),
    Output("cell-alpha", "value"),
    Output("cell-beta", "value"),
    Output("cell-gamma", "value"),
    Input("cell-a", "n_blur"),
    Input("cell-b", "n_blur"),
    Input("cell-c", "n_blur"),
    Input("cell-alpha", "n_blur"),
    Input("cell-beta", "n_blur"),
    Input("cell-gamma", "n_blur"),
    Input("clear-btn-a", "n_clicks"),
    Input("clear-btn-b", "n_clicks"),
    Input("clear-btn-c", "n_clicks"),
    Input("clear-btn-alpha", "n_clicks"),
    Input("clear-btn-beta", "n_clicks"),
    Input("clear-btn-gamma", "n_clicks"),
    State("cell-a", "value"),
    State("cell-b", "value"),
    State("cell-c", "value"),
    State("cell-alpha", "value"),
    State("cell-beta", "value"),
    State("cell-gamma", "value"),
    prevent_initial_call=True
)
def format_number_all(_a, _b, _c, _alpha, _beta, _gamma, n_clicks_a, n_clicks_b, n_clicks_c, n_clicks_alpha, n_click_beta, n_clicks_gamma, value_a, value_b, value_c, value_alpha, value_beta, value_gamma):

    ctx = dash.callback_context

    if not ctx.triggered:
        return value_a, value_b, value_c, value_alpha, value_beta, value_gamma

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    def format_value(value):
        if value is not None and value != "":
            try:
                output_value = f"{float(value):.4f}" #np.around(value, 4)  # Automatically rounds to 4 decimal places
                return output_value
            except ValueError:
                return ""
        return value

    if trigger_id == "clear-btn-a":
        return "", value_b, value_c, value_alpha, value_beta, value_gamma
    if trigger_id == "clear-btn-b":
        return value_a, "", value_c, value_alpha, value_beta, value_gamma
    if trigger_id == "clear-btn-c":
        return value_a, value_b, "", value_alpha, value_beta, value_gamma
    if trigger_id == "clear-btn-alpha":
        return value_a, value_b, value_c, "", value_beta, value_gamma
    if trigger_id == "clear-btn-beta":
        return value_a, value_b, value_c, value_alpha, "", value_gamma
    if trigger_id == "clear-btn-gamma":
        return value_a, value_b, value_c, value_alpha, value_beta, ""

    return format_value(value_a), format_value(value_b), format_value(value_c), format_value(value_alpha), format_value(value_beta), format_value(value_gamma)


@app.callback(
    Output("inactive-elements-store", "data"),
    [
        Input({"type": "ptable-cell", "element": dash.ALL}, "n_clicks"),
        Input("select-all", "n_clicks"),
        Input("unselect-all", "n_clicks"),
    ],
    State("inactive-elements-store", "data"),
    prevent_initial_call=True
)
def update_element_style(n_clicks_all, select_all, unselect_all, current_data):
    trigger = ctx.triggered_id
    updated_data = current_data.copy()

    if trigger == "select-all":
        updated_data = {key: False for key in updated_data}  # Activate all
    elif trigger == "unselect-all":
        updated_data = {key: True for key in updated_data}  # Deactivate all
    elif isinstance(trigger, dict) and trigger.get("type") == "ptable-cell":
        element = trigger["element"]
        updated_data[element] = not updated_data[element]  # Toggle

    return updated_data  # Only update store, frontend will handle styles

# Clientside Callback: Updates element styles instantly
app.clientside_callback(
    """
    function(updateDict, elementToGroup, groupColors) {
        let styles = [];

        for (let el in updateDict) {
            let group = elementToGroup[el] || "nonmetal";
            let baseColor = groupColors[group] || "#4CAF50";
            let style = {
                "width": "30px",
                "height": "30px",
                "margin": "0px",
                "border": "0px solid #ccc",
                "borderRadius": "0px",
                "cursor": "pointer",
                "fontSize": "14px",
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            };

            if (updateDict[el]) {
                // Inactive (gray)
                style["backgroundColor"] = "rgba(238, 238, 238, 0.1)";
                style["color"] = "#666666";
            } else {
                // Active (group color)
                style["backgroundColor"] = baseColor;
                style["color"] = "black";
            }
            styles.push(style);
        }
        return styles;
    }
    """,
    Output({"type": "ptable-cell", "element": dash.ALL}, "style"),
    Input("inactive-elements-store", "data"),
    Input("element-to-group-store", "data"),
    Input("group-colors-store", "data"),
)

@app.callback(
    Output("cif-string-container", "children"),
    Output(structure_component.id(), "data"),
    Output("crystal-vis-container-pxrd-plot", "figure"),
    Output("crystal-vis-container-pxrd-plot", "style"),
    Output("loading-div", "children"),
    Output("error-div", "children"),
    Output("file-name-display-cif", "children"),
    Output("file-name-display-pxrd", "children"),
    Input("upload-cif", "contents"),
    Input("upload-cif", "filename"),
    Input("upload-pxrd", "contents"),
    Input("upload-pxrd", "filename"),
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
    State("inactive-elements-store", "data"),
)
def generate_structures(
    cif_content,
    cif_name,
    pxrd_content,
    pxrd_name,
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
    inactive_elements
):
        
    display_cif_name = f"{cif_name}" if cif_name else "no file uploaded"
    display_pxrd_name = f"{pxrd_name}" if pxrd_name else "no file uploaded"
        
    # Start with an empty figure
    fig = make_subplots(rows=1, cols=1)
    fig_style = {"display": "none"}

    # 1) Check for uploaded CIF to build a reference
    cond_vec = None
    pxrd_ref = None

    if cif_content and cif_name:
        _, cif_content_byte = cif_content.split(',')
        cif_string = base64.b64decode(cif_content_byte).decode("utf-8")

        # Build reference XRD
        #structure = Structure.from_string(cif_string)
        #cif_string = CifWriter(struct=structure, symprec=0.1).__str__()
        pxrd = generate_continuous_xrd_from_cif(cif_string, qmin=0.0, qmax=10.0, debug=True, fwhm_range=(FWHM, FWHM))
        if pxrd is not None:
            # Example: condition vector from the PXRD
            cond_vec = torch.from_numpy(pxrd['iq']).unsqueeze(0).to('cuda')

        # Also create a smaller-range PXRD for display
        pxrd_ref = generate_continuous_xrd_from_cif(cif_string, qmin=0.5, qmax=7.9, qstep=0.005, debug=True, fwhm_range=(FWHM, FWHM))
        if pxrd_ref is not None:
            fig.add_trace(go.Scatter(x=pxrd_ref['q'], y=pxrd_ref['iq'], mode='lines', name='Reference'))
            fig.update_layout(
                xaxis_title=r'Q [Å<sup>-1</sup>]',  # Proper subscript formatting
                yaxis_title='I(Q) [a.u.]',
                height=400,
                width=475,
                yaxis=dict(
                    tickvals=[], 
                    showgrid=False, 
                    zeroline=False
                ),
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    mirror=True, 
                    linewidth=1, 
                    linecolor='black'
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=14),
                legend=dict(
                    x=0.6, 
                    y=1.0, 
                )
            )
            fig_style = {}
        else:
            raise Exception("Cannot generate PXRD from given CIF")

    elif pxrd_content and pxrd_name:
        # If a standalone PXRD was uploaded, you could parse that here.
        # For brevity, we skip it.
        pass
    else:
        pass

    if ctx.triggered_id == "generate-button":

        # 2) Load your model
        #    For demonstration, we’ll just pick one path:
        if cond_vec is not None:
            model_path = "../../../phd_projects/deCIFer/experiments/model__conditioned_mlp_augmentation__context_3076__robust_full_trainingcurves/ckpt.pt"
        else:
            model_path = "../../../phd_projects/deCIFer/experiments/model__nocond__context_3076__robust_full_trainingcurves/ckpt.pt"

        model = load_model_from_checkpoint(model_path, device='cuda')

        # 3) Build a simple atoms string list
        atoms = []
        for i, (el, mtpl, x, y, z) in enumerate(zip(atoms_element, atoms_mult, atoms_x, atoms_y, atoms_z)):
            # Make sure no None in the line
            if None not in [el, mtpl, x, y, z]:
                atoms.append(f"{el} {el}{i} {int(mtpl)} {x:.4f} {y:.4f} {z:.4f} 1.0000")

        # 4) Actually generate the CIF using a hypothetical model method
        #    You would replace with your real generate_custom or similar
        generated = model.generate_custom(
            idx=torch.tensor([START_ID]).unsqueeze(0).to('cuda'),
            max_new_tokens=3076,
            cond_vec=cond_vec,
            start_indices_batch=[[0]],
            composition_string=composition_string if composition_string != "" else None,
            spacegroup_string=spacegroup_string,
            cell_a_string=f'{cell_a_value:.4f}' if cell_a_value else None,
            cell_b_string=f'{cell_b_value:.4f}' if cell_b_value else None,
            cell_c_string=f'{cell_c_value:.4f}' if cell_c_value else None,
            cell_alpha_string=f'{cell_alpha_value:.4f}' if cell_alpha_value else None,
            cell_beta_string=f'{cell_beta_value:.4f}' if cell_beta_value else None,
            cell_gamma_string=f'{cell_gamma_value:.4f}' if cell_gamma_value else None,
            atoms_string_list=atoms if len(atoms) > 0 else None,
            exclude_elements=[el for (el,inact) in inactive_elements.items() if inact],
        ).cpu().numpy()

        # Remove padding
        generated = [ids[ids != PADDING_ID] for ids in generated]

        cif_string_gen = DECODE(generated[0])
        # Fix P1 issues if needed
        cif_string_gen = replace_symmetry_loop_with_P1(cif_string_gen)
        spacegroup_symbol = extract_space_group_symbol(cif_string_gen)
        if spacegroup_symbol != "P 1":
            cif_string_gen = reinstate_symmetry_loop(cif_string_gen, spacegroup_symbol)

        # 5) Generate XRD from the newly created CIF
        pxrd_gen = generate_continuous_xrd_from_cif(cif_string_gen, qmin=0.5, qmax=7.9, qstep=0.005, debug=True, fwhm_range=(FWHM, FWHM))
        if pxrd_gen is not None:
            fig.add_trace(go.Scatter(x=pxrd_gen['q'], y=pxrd_gen['iq'], mode='lines', name='Generated'))
            if pxrd_ref is not None:
                diff = pxrd_ref['iq'] - pxrd_gen['iq'] - 0.1
                rwp_val = rwp_fn(pxrd_ref['iq'], pxrd_gen['iq'])
                fig.add_trace(go.Scatter(
                    x=pxrd_gen['q'], y=diff, mode='lines',
                    name=f'Difference, Rwp: {rwp_val:.3f}'
                ))
            fig.update_layout(
                xaxis_title=r'Q [Å<sup>-1</sup>]',  # Proper subscript formatting
                yaxis_title='I(Q) [a.u.]',
                height=400,
                width=475,
                yaxis=dict(
                    tickvals=[], 
                    showgrid=False, 
                    zeroline=False
                ),
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    mirror=True, 
                    linewidth=1, 
                    linecolor='black'
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=14),
                legend=dict(
                    x=0.6, 
                    y=1.0, 
                    #bgcolor='rgba(255,255,255,0.5)',
                    #bordercolor='black', 
                    #borderwidth=1
                )
            )

        # 6) Generate display and structure data
        cif_display = html.Pre(cif_string_gen, style={
            'whiteSpace': 'pre-wrap',
            'maxHeight': '950px',
            'overflow': 'auto',
            'backgroundColor': '#f8f9fa',
            'padding': '10px',
            'border': '1px solid #ddd'
        })

        structure_gen = Structure.from_str(cif_string_gen, fmt='cif')

        return (
            cif_display,
            structure_gen,
            fig,
            {},  # style for the figure
            "",
            html.Div(),  # no errors
            display_cif_name,
            display_pxrd_name,
        )
    else:
        # If no generation triggered yet, return empty placeholders
        return (
            html.Div(),
            None,
            fig,
            fig_style,
            "",
            html.Div(),
            display_cif_name,
            display_pxrd_name,
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
        dash.dcc.Input(id={"type": "atom-element", "index": new_index}, type="text",
                       placeholder="Element", style={"width": "15%", "marginRight": "5px"}),
        dash.dcc.Input(id={"type": "atom-multiplicity", "index": new_index}, type="number",
                       placeholder="Multiplicity", style={"width": "15%", "marginRight": "5px"}),
        dash.dcc.Input(id={"type": "atom-x", "index": new_index}, type="number",
                       placeholder="x", style={"width": "15%", "marginRight": "5px"}),
        dash.dcc.Input(id={"type": "atom-y", "index": new_index}, type="number",
                       placeholder="y", style={"width": "15%", "marginRight": "5px"}),
        dash.dcc.Input(id={"type": "atom-z", "index": new_index}, type="number",
                       placeholder="z", style={"width": "15%", "marginRight": "5px"}),
        dash.dcc.Input(value="1.0", disabled=True, type="number", placeholder="Occupancy",
                       style={"width": "5%"})
    ], style={"marginBottom": "5px"})
    children.append(new_atom)
    return children

# Use Crystal Toolkit's layout registration
ctc.register_crystal_toolkit(app, layout=layout)


if __name__ == "__main__":

    # Run server
    app.run_server(debug=True, port=8060)
