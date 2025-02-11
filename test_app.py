import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pymatgen.core import Structure
import crystal_toolkit.components as ctc
from crystal_toolkit.components import StructureMoleculeComponent
from crystal_toolkit.components.diffraction import XRayDiffractionComponent
from crystal_toolkit.helpers.layouts import Container
from pymatgen.core.structure import Lattice

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define a sample structure (Silicon in this case)
structure = Structure.from_file("test_cif.cif")
structure_component = StructureMoleculeComponent(structure)
xrd_component = XRayDiffractionComponent(initial_structure=structure)
print(xrd_component)

# Layout of the Dash app
layout = app.layout = html.Div([
    html.H1("Crystal Toolkit X-Ray Diffraction"),
    html.Div([structure_component.layout()]),
    html.Div([xrd_component.layout()]),
])

# Register callbacks for interactivity
ctc.register_crystal_toolkit(app, layout=layout)
#xrd_component.generate_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True, port=8060)
