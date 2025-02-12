from dash import Dash, dcc, html, Input, Output, State, ctx
import dash

# Constants for styling
PERIODIC_DIM = "40px"  # Define the periodic table cell size
PERIODIC_FONTSIZE = "14px"

# Group colors for element types
group_colors = {
    "nonmetal": "#4CAF50",
    "alkali metal": "#FF5722",
    "alkaline earth metal": "#FF9800",
    "transition metal": "#9C27B0",
}

# Mapping of elements to their groups
element_to_group = {
    "H": "nonmetal", "He": "nonmetal", "Li": "alkali metal",
    "Be": "alkaline earth metal", "B": "nonmetal", "C": "nonmetal",
    "N": "nonmetal", "O": "nonmetal", "F": "nonmetal", "Ne": "nonmetal"
}

# Define initial state for element activity
inactive_elements_dict = {el: False for el in element_to_group.keys()}

app = Dash(__name__)

app.layout = html.Div([
    # Store to hold inactive elements
    dcc.Store(id="inactive-elements-store", data=inactive_elements_dict),

    # Buttons to select/unselect all elements
    html.Button("Select All", id="select-all", n_clicks=0),
    html.Button("Unselect All", id="unselect-all", n_clicks=0),

    # Periodic table elements as buttons
    html.Div([
        html.Button(
            el,
            id={"type": "ptable-cell", "element": el},
            n_clicks=0,
            style={
                "width": PERIODIC_DIM,
                "height": PERIODIC_DIM,
                "backgroundColor": group_colors.get(element_to_group[el], "#4CAF50"),
                "color": "black",
                "cursor": "pointer",
                "fontSize": PERIODIC_FONTSIZE,
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
            }
        ) for el in inactive_elements_dict
    ]),
])

# 🔥 Clientside Callback (Instant UI Updates)
app.clientside_callback(
    """
    function(triggeredId, currentState) {
        let updateDict = { ...currentState };  // Clone state
        let styles = [];

        let groupColors = {
            "nonmetal": "#4CAF50",
            "alkali metal": "#FF5722",
            "alkaline earth metal": "#FF9800",
            "transition metal": "#9C27B0"
        };
        let elementToGroup = {
            "H": "nonmetal", "He": "nonmetal", "Li": "alkali metal",
            "Be": "alkaline earth metal", "B": "nonmetal", "C": "nonmetal",
            "N": "nonmetal", "O": "nonmetal", "F": "nonmetal", "Ne": "nonmetal"
        };

        if (triggeredId === "select-all") {
            // Activate all elements instantly
            for (let el in updateDict) updateDict[el] = false;
        } else if (triggeredId === "unselect-all") {
            // Deactivate all elements instantly
            for (let el in updateDict) updateDict[el] = true;
        } else if (typeof triggeredId === "object" && triggeredId.type === "ptable-cell") {
            let element = triggeredId.element;
            updateDict[element] = !updateDict[element];  // Toggle state
        }

        // Generate new styles
        for (let el in updateDict) {
            let group = elementToGroup[el] || "nonmetal";
            let baseColor = groupColors[group] || "#4CAF50";

            let style = {
                "width": "40px",
                "height": "40px",
                "margin": "0px",
                "border": "0px solid #ccc",
                "borderRadius": "0px",
                "cursor": "pointer",
                "fontSize": "14px",
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "backgroundColor": updateDict[el] ? "rgba(238, 238, 238, 0.1)" : baseColor,
                "color": updateDict[el] ? "#666666" : "black"
            };
            styles.push(style);
        }

        return [updateDict, styles];  // Update store + UI instantly
    }
    """,
    [Output("inactive-elements-store", "data"), Output({"type": "ptable-cell", "element": dash.ALL}, "style")],
    [Input({"type": "ptable-cell", "element": dash.ALL}, "n_clicks"),
     Input("select-all", "n_clicks"),
     Input("unselect-all", "n_clicks")],
    [State("inactive-elements-store", "data")]
)

# 🐢 Backend Python Callback (Processes Updates Later)
@app.callback(
    Output("inactive-elements-store", "data"),
    [
        Input("inactive-elements-store", "data")  # Detects changes
    ],
    prevent_initial_call=True
)
def backend_process(updated_data):
    # Here, we could process and store data elsewhere (DB, logs, etc.)
    print("Backend processing:", updated_data)
    return updated_data  # This ensures state consistency
if __name__ == "__main__":
    app.run_server(debug=True, port=8090)
