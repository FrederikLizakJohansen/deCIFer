# app.py
import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Button(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Span("Click Me!")
                            ],
                            className="button-inner"
                        )
                    ],
                    className="button-outer"
                )
            ],
            id="styled-button"
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True, port=8070)
