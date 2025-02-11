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

# Instantiate the app
app = Dash(assets_folder=SETTINGS.ASSETS_PATH)
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
all_elements = []
for row in periodic_table_layout:
    for el in row:
        if el is not None and el not in all_elements:
            all_elements.append(el)

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

PERIODIC_DIM = "20px"
PERIODIC_FONTSIZE = "0.6em"

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
                            "margin": "2px",
                            "border": "1px solid #ccc",
                            "borderRadius": "4px",
                            "backgroundColor": base_color,
                            "color": "white",
                            "cursor": "pointer",
                            "fontSize": PERIODIC_FONTSIZE
                        }
                    )
                ))
        table_rows.append(html.Tr(cells))
    return html.Table(table_rows, style={"borderCollapse": "collapse", "margin": "0 auto", "width": "auto"})

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

    # Top row with three columns
    html.Div([
        # Left column: input forms
        html.Div([
            html.H3("Structure Generation Input", style={
                "textAlign": "center", 
                "fontFamily": 'Helvetica Neue, sans-serif', 
                "fontSize": "20px", 
                "margin-bottom": "10px"
            }),
            # ---- New Periodic Table Section Start ----
            dcc.Store(id="inactive-elements-store", data=[]),
            dcc.Store(id="selected-elements-store", data=all_elements),
            dcc.Store(id="ptable-states", data={el: True for el in all_elements}),
            html.Div([
                html.H3("Periodic Table", style={"textAlign": "center", "marginBottom": "10px"}),
                create_periodic_table(),
                html.Div([
                    html.Button("Select All", id="select-all", n_clicks=0, style={"marginRight": "5px"}),
                    html.Button("Unselect All", id="unselect-all", n_clicks=0)
                ], style={"marginTop": "10px", "textAlign": "center"})
            ], style={
                "marginBottom": "20px",
                "padding": "10px",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
                "backgroundColor": "#ffffff"
            }),

            html.Div([
                html.Div([
                    dcc.Upload(
                        id="upload-cif",
                        children=html.Div(["🎉 Upload CIF"]),
                        multiple=False, 
                        style={
                            "width": "200px",
                            "backgroundColor": "#FF6B6B",
                            "color": "white",
                            "borderRadius": "8px",
                            "lineHeight": "30px",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "padding": "5px 10px",
                            "minHeight": "15px",
                            "boxShadow": "2px 2px 8px rgba(0, 0, 0, 0.3)",
                            "transition": "all 0.2s ease-in-out",
                        },
                        style_active={
                            "backgroundColor": "#FF8A65",
                            "boxShadow": "0 0 10px rgba(0,0,0,0.4)",
                            "transform": "scale(1.05)",
                        },
                        accept=".cif"
                    ),
                    dcc.Upload(
                        id="upload-pxrd",
                        children=html.Div(["💥 Upload PXRD"]),
                        multiple=False,
                        style={
                            "width": "200px",
                            "backgroundColor": "#6BCBFF",
                            "color": "white",
                            "borderRadius": "8px",
                            "lineHeight": "30px",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "padding": "5px 10px",
                            "minHeight": "15px",
                            "boxShadow": "2px 2px 8px rgba(0, 0, 0, 0.3)",
                            "transition": "all 0.2s ease-in-out",
                        },
                        style_active={
                            "backgroundColor": "#42A5F5",
                            "boxShadow": "0 0 10px rgba(0,0,0,0.4)",
                            "transform": "scale(1.05)",
                        },
                        accept=".xy"
                    ),
                ], style={
                    "marginBottom": "10px",
                    "display": "flex",
                    "justifyContent": "space-evenly"
                }),

                html.Div([
                    html.Label("Exact Composition:"),
                    dcc.Input(id="composition-input", type="text", placeholder="e.g. A2BX6", style={"width": "100%"})
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("Space Group:"),
                    dcc.Dropdown(
                        id="spacegroup-dropdown",
                        options=spacegroup_dropdown,
                        placeholder="Select a space group",
                        value="None",
                    )
                ], style={"marginBottom": "10px"}),

                # Cell parameters
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
                                    style={"width": "15%"}
                                )
                            ], style={"marginBottom": "5px"})
                        ]
                    ),
                    html.Button("Add Atom", id="add-atom-button", n_clicks=0, style={"marginTop": "5px"})
                ], style={"marginBottom": "20px"}),

                # Generate button + loading
                html.Div([
                    html.Button(
                        "🚀 Generate",
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
                    "gap": "50px",
                    "marginTop": "10px"
                }),

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

        # Middle column: CIF display
        html.Div([
            html.H3("CIF Display", style={
                "textAlign": "center",
                "fontFamily": 'Helvetica Neue, sans-serif',
                "fontSize": "20px",
                "margin-bottom": "10px"
            }),
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

        # Right column: crystal + PXRD
        html.Div([
            html.H3("Crystal Display", style={
                "textAlign": "center", 
                "fontFamily": 'Helvetica Neue, sans-serif', 
                "fontSize": "20px", 
                "margin-bottom": "10px"
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
                        "backgroundColor": "white",
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
                        "backgroundColor": "white",
                        "minHeight": "260px",
                        "maxHeight": "260px"
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
])

@app.callback(
    Output("ptable-states", "data"),
    Input("select-all", "n_clicks"),
    Input("unselect-all", "n_clicks"),
    Input({"type": "ptable-cell", "element": dash.ALL}, "n_clicks"),
    State({"type": "ptable-cell", "element": dash.ALL}, "id"),
    State("ptable-states", "data")
)
def update_ptable_states(select_all, unselect_all, cell_n_clicks, cell_ids, current_states):
    trigger = ctx.triggered_id
    # If a global action was triggered, override all cell states.
    if trigger in ["select-all", "unselect-all"]:
        if trigger == "select-all":
            return {comp_id["element"]: True for comp_id in cell_ids}
        elif trigger == "unselect-all":
            return {comp_id["element"]: False for comp_id in cell_ids}
    # Otherwise, update based on individual cell clicks.
    # (Here we use the parity of each cell’s n_clicks: even means active, odd means inactive.)
    new_states = {}
    for n_clicks, comp_id in zip(cell_n_clicks, cell_ids):
        if n_clicks is None:
            n_clicks = 0
        new_states[comp_id["element"]] = (n_clicks % 2 == 0)
    return new_states

@app.callback(
    Output({"type": "ptable-cell", "element": dash.MATCH}, "style"),
    Input("ptable-states", "data"),
    State({"type": "ptable-cell", "element": dash.MATCH}, "id")
)
def update_cell_style(ptable_states, cell_id):
    cell_width = "30px"
    cell_height = "30px"
    element = cell_id["element"]
    group = element_to_group.get(element, "nonmetal")
    base_color = group_colors.get(group, "#4CAF50")
    if ptable_states.get(element, True):
        # Active style: show the element’s group color.
        return {
            "width": cell_width,
            "height": cell_height,
            "margin": "2px",
            "border": "1px solid #ccc",
            "borderRadius": "4px",
            "backgroundColor": base_color,
            "color": "white",
            "cursor": "pointer",
            "fontSize": "0.8em"
        }
    else:
        # Inactive style: show a greyed-out look.
        return {
            "width": cell_width,
            "height": cell_height,
            "margin": "2px",
            "border": "1px solid #ccc",
            "borderRadius": "4px",
            "backgroundColor": "#eeeeee",
            "color": "#666666",
            "cursor": "pointer",
            "fontSize": "0.8em"
        }

# @app.callback(
#     Output({"type": "ptable-cell", "element": dash.MATCH}, "style"),
#     Input({"type": "ptable-cell", "element": dash.MATCH}, "n_clicks"),
#     State({"type": "ptable-cell", "element": dash.MATCH}, "id")
# )
# def toggle_cell_style(n_clicks, cell_id):
#     # Define the reduced dimensions.
#     cell_width = PERIODIC_DIM
#     cell_height = PERIODIC_DIM
#     element = cell_id["element"]
#     group = element_to_group.get(element, "nonmetal")
#     base_color = group_colors.get(group, "#4CAF50")
#     # If n_clicks is even (or None), treat the element as active.
#     if not n_clicks or n_clicks % 2 == 0:
#         return {
#             "width": cell_width,
#             "height": cell_height,
#             "margin": "2px",
#             "border": "1px solid #ccc",
#             "borderRadius": "4px",
#             "backgroundColor": base_color,
#             "color": "white",
#             "cursor": "pointer",
#             "fontSize": PERIODIC_FONTSIZE,
#         }
#     else:
#         # If odd, the element is inactive (greyed out).
#         return {
#             "width": cell_width,
#             "height": cell_height,
#             "margin": "2px",
#             "border": "1px solid #ccc",
#             "borderRadius": "4px",
#             "backgroundColor": "#eeeeee",
#             "color": "#666666",
#             "cursor": "pointer",
#             "fontSize": PERIODIC_FONTSIZE,
#         }
#
# @app.callback(
#     Output("inactive-elements-store", "data"),
#     Input({"type": "ptable-cell", "element": dash.ALL}, "n_clicks"),
#     State({"type": "ptable-cell", "element": dash.ALL}, "id")
# )
# def aggregate_inactive(n_clicks_list, ids):
#     inactive = []
#     for n_clicks, comp_id in zip(n_clicks_list, ids):
#         # Treat a missing n_clicks as 0 (active).
#         if n_clicks is None:
#             n_clicks = 0
#         if n_clicks % 2 == 1:
#             inactive.append(comp_id["element"])
#     return inactive

# @app.callback(
#     Output("selected-elements-store", "data"),
#     Input({"type": "ptable-cell", "element": dash.ALL}, "n_clicks"),
#     Input("select-all", "n_clicks"),
#     Input("unselect-all", "n_clicks"),
#     State("selected-elements-store", "data")
# )
# def update_selected_elements(cell_clicks, select_all, unselect_all, current_selected):
#     trigger = ctx.triggered_id
#     if trigger is None:
#         raise dash.exceptions.PreventUpdate
#     if trigger == "select-all":
#         return all_elements
#     elif trigger == "unselect-all":
#         return []
#     else:
#         # The trigger is one of the periodic table cell buttons (its id is a dict).
#         element = trigger["element"]
#         if element in current_selected:
#             new_selected = [el for el in current_selected if el != element]
#         else:
#             new_selected = current_selected + [element]
#         return new_selected
#
# app.clientside_callback(
#     """
#     function(selectedElements) {
#         // Define the cell dimensions.
#         var cell_width = "30px";
#         var cell_height = "30px";
#         // Define the group colors.
#         var group_colors = {
#             "alkali": "#FF6666",         // red-ish
#             "alkaline": "#FFB347",       // orange-ish
#             "transition": "#B0C4DE",     // light steel blue
#             "post-transition": "#CCCCCC",// grey
#             "metalloid": "#CCCC99",      // olive-ish
#             "nonmetal": "#90EE90",       // light green
#             "halogen": "#66CDAA",        // medium aquamarine
#             "noble": "#87CEFA",          // light sky blue
#             "lanthanide": "#FFB6C1",     // light pink
#             "actinide": "#FFA07A"        // light salmon
#         };
#         // Define element to group mapping.
#         var element_to_group = {
#             "H": "nonmetal", "He": "noble", "Li": "alkali", "Be": "alkaline",
#             "B": "metalloid", "C": "nonmetal", "N": "nonmetal", "O": "nonmetal",
#             "F": "halogen", "Ne": "noble", "Na": "alkali", "Mg": "alkaline",
#             "Al": "post-transition", "Si": "metalloid", "P": "nonmetal",
#             "S": "nonmetal", "Cl": "halogen", "Ar": "noble", "K": "alkali",
#             "Ca": "alkaline", "Sc": "transition", "Ti": "transition", "V": "transition",
#             "Cr": "transition", "Mn": "transition", "Fe": "transition", "Co": "transition",
#             "Ni": "transition", "Cu": "transition", "Zn": "transition", "Ga": "post-transition",
#             "Ge": "metalloid", "As": "metalloid", "Se": "nonmetal", "Br": "halogen",
#             "Kr": "noble", "Rb": "alkali", "Sr": "alkaline", "Y": "transition",
#             "Zr": "transition", "Nb": "transition", "Mo": "transition", "Tc": "transition",
#             "Ru": "transition", "Rh": "transition", "Pd": "transition", "Ag": "transition",
#             "Cd": "transition", "In": "post-transition", "Sn": "post-transition",
#             "Sb": "metalloid", "Te": "metalloid", "I": "halogen", "Xe": "noble",
#             "Cs": "alkali", "Ba": "alkaline", "La": "lanthanide", "Hf": "transition",
#             "Ta": "transition", "W": "transition", "Re": "transition", "Os": "transition",
#             "Ir": "transition", "Pt": "transition", "Au": "transition", "Hg": "transition",
#             "Tl": "post-transition", "Pb": "post-transition", "Bi": "post-transition",
#             "Po": "metalloid", "At": "halogen", "Rn": "noble", "Fr": "alkali",
#             "Ra": "alkaline", "Rf": "transition", "Db": "transition", "Sg": "transition",
#             "Bh": "transition", "Hs": "transition", "Mt": "transition", "Ds": "transition",
#             "Rg": "transition", "Cn": "transition", "Nh": "post-transition", "Fl": "post-transition",
#             "Mc": "post-transition", "Lv": "post-transition", "Ts": "halogen", "Og": "noble",
#             "Ce": "lanthanide", "Pr": "lanthanide", "Nd": "lanthanide", "Pm": "lanthanide",
#             "Sm": "lanthanide", "Eu": "lanthanide", "Gd": "lanthanide", "Tb": "lanthanide",
#             "Dy": "lanthanide", "Ho": "lanthanide", "Er": "lanthanide", "Tm": "lanthanide",
#             "Yb": "lanthanide", "Lu": "lanthanide", "Ac": "actinide", "Th": "actinide",
#             "Pa": "actinide", "U": "actinide", "Np": "actinide", "Pu": "actinide",
#             "Am": "actinide", "Cm": "actinide", "Bk": "actinide", "Cf": "actinide",
#             "Es": "actinide", "Fm": "actinide", "Md": "actinide", "No": "actinide",
#             "Lr": "actinide"
#         };
#         // The layout array must mirror the order of non-null elements created in our table.
#         var periodic_table_layout = [
#             ["H", null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, "He"],
#             ["Li", "Be", null, null, null, null, null, null, null, null, null, null, "B", "C", "N", "O", "F", "Ne"],
#             ["Na", "Mg", null, null, null, null, null, null, null, null, null, null, "Al", "Si", "P", "S", "Cl", "Ar"],
#             ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
#             ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
#             ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", null],
#             ["Fr", "Ra", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", null],
#             [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null],
#             [null, null, null, "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", null],
#             [null, null, "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", null]
#         ];
#         var styles = [];
#         // Iterate through the layout in the same order as the table built.
#         for (var i = 0; i < periodic_table_layout.length; i++) {
#             for (var j = 0; j < periodic_table_layout[i].length; j++) {
#                 var cell = periodic_table_layout[i][j];
#                 if (cell !== null) {
#                     var group = element_to_group[cell] || "nonmetal";
#                     var base_color = group_colors[group] || "#4CAF50";
#                     var style;
#                     if (selectedElements && selectedElements.indexOf(cell) !== -1) {
#                         // If the element is selected, use its base (group) color.
#                         style = {
#                             width: cell_width,
#                             height: cell_height,
#                             margin: "2px",
#                             border: "1px solid #ccc",
#                             borderRadius: "4px",
#                             backgroundColor: base_color,
#                             color: "white",
#                             cursor: "pointer",
#                             fontSize: "0.8em"
#                         };
#                     } else {
#                         // Otherwise, show a greyed-out style.
#                         style = {
#                             width: cell_width,
#                             height: cell_height,
#                             margin: "2px",
#                             border: "1px solid #ccc",
#                             borderRadius: "4px",
#                             backgroundColor: "#eeeeee",
#                             color: "#666666",
#                             cursor: "pointer",
#                             fontSize: "0.8em"
#                         };
#                     }
#                     styles.push(style);
#                 }
#             }
#         }
#         return styles;
#     }
#     """,
#     Output({"type": "ptable-cell", "element": dash.ALL}, "style"),
#     Input("selected-elements-store", "data")
# );

# @app.callback(
#     Output({"type": "ptable-cell", "element": dash.ALL}, "style"),
#     Input("selected-elements-store", "data")
# )
# def update_ptable_styles(selected_elements):
#     styles = []
#     # New dimensions: make the table less wide.
#     cell_width = "20px"
#     cell_height = "20px"
#     for row in periodic_table_layout:
#         for cell in row:
#             if cell is not None:
#                 # Look up the element's chemical group and corresponding base color.
#                 group = element_to_group.get(cell, "nonmetal")
#                 base_color = group_colors.get(group, "#4CAF50")
#                 if cell in selected_elements:
#                     # If the element is selected, use its group color.
#                     style = {
#                         "width": cell_width,
#                         "height": cell_height,
#                         "margin": "2px",
#                         "border": "1px solid #ccc",
#                         "borderRadius": "4px",
#                         "backgroundColor": base_color,
#                         "color": "white",
#                         "cursor": "pointer",
#                         "fontSize": "0.6em"
#                     }
#                 else:
#                     # If unselected, show a greyed-out style.
#                     style = {
#                         "width": cell_width,
#                         "height": cell_height,
#                         "margin": "2px",
#                         "border": "1px solid #ccc",
#                         "borderRadius": "4px",
#                         "backgroundColor": "#eeeeee",
#                         "color": "#666666",
#                         "cursor": "pointer",
#                         "fontSize": "0.6em"
#                     }
#                 styles.append(style)
#     return styles


@app.callback(
    Output("cif-string-container", "children"),
    Output(structure_component.id(), "data"),
    Output("crystal-vis-container-pxrd-plot", "figure"),
    Output("crystal-vis-container-pxrd-plot", "style"),
    Output("loading-div", "children"),
    Output("error-div", "children"),
    Input("upload-cif", "contents"),
    Input("upload-cif", "filename"),
    Input("upload-pxrd", "contents"),
    Input("upload-pxrd", "filename"),
    #Input("upload-cif", "isCompleted"),
    #Input("upload-cif", "fileNames"),
    #Input("upload-pxrd", "isCompleted"),
    #Input("upload-pxrd", "fileNames"),
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
):

    if ctx.triggered_id == "generate-button":
        # Start with an empty figure
        fig = make_subplots(rows=1, cols=1)

        # 1) Check for uploaded CIF to build a reference
        cond_vec = None
        pxrd_ref = None

        if cif_content and cif_name:
            _, cif_content_byte = cif_content.split(',')
            cif_string = base64.b64decode(cif_content_byte).decode("utf-8")

            # Build reference XRD
            #structure = Structure.from_string(cif_string)
            #cif_string = CifWriter(struct=structure, symprec=0.1).__str__()
            pxrd = generate_continuous_xrd_from_cif(cif_string, qmin=0.0, qmax=10.0, debug=True, fwhm_range=(0.05, 0.05))
            if pxrd is not None:
                # Example: condition vector from the PXRD
                cond_vec = torch.from_numpy(pxrd['iq']).unsqueeze(0).to('cuda')

            # Also create a smaller-range PXRD for display
            pxrd_ref = generate_continuous_xrd_from_cif(cif_string, qmin=0.5, qmax=7.5, debug=True, fwhm_range=(0.05, 0.05))
            if pxrd_ref is not None:
                fig.add_trace(go.Scatter(x=pxrd_ref['q'], y=pxrd_ref['iq'], mode='lines', name='Reference'))
                fig.update_layout(
                   xaxis_title='Q [Å^-1]',
                   yaxis_title='I(Q) [a.u.]',
                   height=225,
                   width=475,
                   yaxis=dict(tickvals=[]),
                   margin=dict(l=0, r=0, t=0, b=0),
                   plot_bgcolor='rgba(0, 0, 0, 0)',
                   paper_bgcolor='rgba(0, 0, 0, 0)',
                   legend=dict(x=0.8, y=1.0),
                )
            else:
                raise Exception("Cannot generate PXRD from given CIF")

        elif pxrd_content and pxrd_name:
            # If a standalone PXRD was uploaded, you could parse that here.
            # For brevity, we skip it.
            pass
        else:
            pass

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
            composition_string=composition_string,
            spacegroup_string=spacegroup_string,
            cell_a_string=f'{cell_a_value:.4f}' if cell_a_value else None,
            cell_b_string=f'{cell_b_value:.4f}' if cell_b_value else None,
            cell_c_string=f'{cell_c_value:.4f}' if cell_c_value else None,
            cell_alpha_string=f'{cell_alpha_value:.4f}' if cell_alpha_value else None,
            cell_beta_string=f'{cell_beta_value:.4f}' if cell_beta_value else None,
            cell_gamma_string=f'{cell_gamma_value:.4f}' if cell_gamma_value else None,
            atoms_string_list=atoms if len(atoms) > 0 else None,
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
        pxrd_gen = generate_continuous_xrd_from_cif(cif_string_gen, qmin=0.5, qmax=7.5, debug=True, fwhm_range=(0.05, 0.05))
        if pxrd_gen is not None:
            fig.add_trace(go.Scatter(x=pxrd_gen['q'], y=pxrd_gen['iq'], mode='lines', name='Generated'))
            if pxrd_ref is not None:
                diff = pxrd_ref['iq'] - pxrd_gen['iq'] - 0.5
                rwp_val = rwp_fn(pxrd_ref['iq'], pxrd_gen['iq'])
                fig.add_trace(go.Scatter(
                    x=pxrd_gen['q'], y=diff, mode='lines',
                    name=f'Difference, Rwp: {rwp_val:.3f}'
                ))
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

        # 6) Generate display and structure data
        cif_display = html.Pre(cif_string_gen, style={
            'whiteSpace': 'pre-wrap',
            'maxHeight': '775px',
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
            html.Div()  # no errors
        )

    # If no generation triggered yet, return empty placeholders
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
                       style={"width": "15%"})
    ], style={"marginBottom": "5px"})
    children.append(new_atom)
    return children

# Use Crystal Toolkit's layout registration
ctc.register_crystal_toolkit(app, layout=layout)


if __name__ == "__main__":

    # Run server
    app.run_server(debug=True, port=8060)
