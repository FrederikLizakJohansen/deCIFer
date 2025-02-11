import dash
from dash import dcc, html, Input, Output, ctx
import time

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Button("Start Process", id="start-button", n_clicks=0),
    dcc.Loading(
        id="loading-indicator",
        type="circle",  # Options: "default", "circle", or "dot"
        children=[html.Div(id="output", children="Click the button to start")]
    ),
])

@app.callback(
    Output("output", "children"),
    Input("start-button", "n_clicks"),
    #prevent_initial_call=True
)
def run_process(n_clicks):
    # Check which component triggered the callback
    if ctx.triggered_id == "start-button":
        time.sleep(3)  # Simulate a long process
        return "Process Completed!"
    return "Waiting for process..."

if __name__ == "__main__":
    app.run_server(debug=True)
