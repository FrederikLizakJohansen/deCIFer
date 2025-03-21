import dash
from dash import dcc, html, Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Button("Click Me", id="btn", n_clicks=0, style={"background-color": "blue", "color": "white", "opacity": "1"}),
    dcc.Interval(id="interval", interval=3000, n_intervals=0, disabled=True),  # 3s interval
])

@app.callback(
    Output("btn", "disabled"),
    Output("btn", "style"),
    Output("interval", "disabled"),
    Input("btn", "n_clicks"),
    prevent_initial_call=True
)
def disable_button(n_clicks):
    disabled_style = {"background-color": "gray", "color": "white", "opacity": "0.5", "cursor": "not-allowed"}
    return True, disabled_style, False  # Disable button, update style, enable interval

@app.callback(
    Output("btn", "disabled", allow_duplicate=True),
    Output("btn", "style", allow_duplicate=True),
    Output("interval", "disabled", allow_duplicate=True),
    Input("interval", "n_intervals"),
    prevent_initial_call=True
)
def enable_button(n_intervals):
    enabled_style = {"background-color": "blue", "color": "white", "opacity": "1", "cursor": "pointer"}
    return False, enabled_style, True  # Re-enable button, reset style, stop interval

if __name__ == "__main__":
    app.run_server(debug=True)
