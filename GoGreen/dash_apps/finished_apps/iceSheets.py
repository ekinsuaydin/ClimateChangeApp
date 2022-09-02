import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from django_plotly_dash import DjangoDash
import pandas as pd
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("iceSheets", external_stylesheets=external_stylesheets)

df = pd.read_excel('GoGreen/static/Media/data/2485_Sept_Arctic_extent_1979-2021.xlsx', engine='openpyxl')
ocean_heat = pd.read_csv('GoGreen/static/Media/data/pent_h22-w0-2000m.dat.txt', header=0, delimiter='\s+')
ocean_heat['Year'] = (ocean_heat['YEAR'].astype(str).str[:4]).astype(int)

oceanheat_seaice = ocean_heat.merge(df, how='inner', left_on='Year', right_on='year')

user_data = {"x": [], "y": []}
df2 = pd.DataFrame(user_data)


def addList(x, y):
    user_data["x"].append(x)
    user_data["y"].append(y)
    df2 = pd.DataFrame(user_data)
    return df2


app.layout = html.Div([
    html.Div([
        html.Div(
            html.H6('Arctic Sea Ice Extent', style={'color': '#F2F3F4'}),
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div(
            daq.BooleanSwitch(id="prediction_line", on=False, color="#515A5A"),
            style={'float': 'right', 'display': 'inline-block'}
        ),

    ]),
    dcc.Graph(id='sea_ice_graph'),
    html.Br(),
    html.P('Which year would you like to predict?', style={'color': '#F2F3F4'}),
    html.Div(
        dcc.Input(id="prediction_year",
                  type="number",
                  min=2020,
                  value=2030),
        style={'display': 'inline-block'}),
    html.Div(
        html.P(id="prediction"),
        style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'}),
    html.Br(),
    html.H5("Ocean Heat Rises, Arctic Sea Ice Extent Descreasing! ", style={'color': '#F2F3F4'}),
    html.Div(
        daq.BooleanSwitch(id="prediction_line2", on=False, color="#515A5A"),
        style={'display': 'inline-block'}),

    dcc.Graph(id='oceanheat_iceextent'),
    html.Br(),
    html.P('Enter ocean heat value(zettajoules)', style={'color': '#F2F3F4'}),
    html.Div(
        dcc.Input(id="heat_value",
                  type="number",
                  min=0,
                  value=15),
        style={'display': 'inline-block'}),
    html.Div(
        html.P(id="prediction2"),
        style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'}),
    html.Br(),
    html.H5("Ocean Heat vs Arctic Sea Ice Graph of Your Predictions", style={'color': '#F2F3F4'}),
    dcc.Graph(id='user_graph'),

], style={'margin-left': '10px'})


@app.callback(
    Output("sea_ice_graph", "figure"),
    Output("prediction", "children"),
    [
        Input("prediction_year", "value"),
        Input("prediction_line", "on")
    ]
)
def generate_chart(prediction_year, on):
    fig = px.line(df, x=df['year'], y=df['extent'])
    fig.update_xaxes(title='Year', showgrid=False)
    fig.update_yaxes(title='Million Square km', showgrid=True, gridwidth=0.05, gridcolor='gray')
    fig.update_layout(paper_bgcolor='black',
                      plot_bgcolor='black',
                      font={"color": "#D3D3D3"}, )

    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig.update_layout(hovermode="x unified")
    fig.update_traces(hovertemplate="<br>".join([
        "%{y} million square km",
        "<extra></extra>"
    ]))

    X = np.array(df['year'])
    y = np.array(df['extent'])
    # Create model for the class to fit in
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    switch = "{}".format(on)
    if switch == "True":
        fig.add_trace(go.Scatter(
            x=X.reshape(-1),
            y=y_predicted,
            line_color='#00FF7F',
            name='Prediction',
            opacity=0.8))

    prediction = poly_reg_model.predict(poly.fit_transform([[prediction_year]]))
    prediction = prediction.flat[0]

    return fig, f'The predicted arctic sea ice extent: \
                {prediction} \
                km'


@app.callback(
    Output("oceanheat_iceextent", "figure"),
    Output("prediction2", "children"),
    Output("user_graph", "figure"),
    [
        Input("heat_value", "value"),
        Input("prediction_line2", "on")
    ]
)
def generate_chart(temp_value, on):
    fig2 = px.scatter(oceanheat_seaice, x=oceanheat_seaice['NH'], y=oceanheat_seaice['extent'])
    fig2.update_yaxes(title='Ocean Heat(zettajoules)', showgrid=False)
    fig2.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Million Square km',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    fig2.update_xaxes(zeroline=False)
    fig2.update_yaxes(zeroline=False)
    fig2.update_traces(hovertemplate="<br>".join([
        "%{x} zettajoules",
        "%{y} ",
        "<extra></extra>"
    ])
    )

    X = np.array(oceanheat_seaice['NH'])
    y = np.array(oceanheat_seaice['extent'])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    switch = "{}".format(on)
    if switch == "True":
        fig2.add_trace(go.Scatter(
            x=X.reshape(-1),
            y=y_predicted,
            line_color='#00FF7F',
            name='Prediction',
            opacity=0.8))

    prediction = poly_reg_model.predict(poly.fit_transform([[temp_value]]))
    prediction = prediction.flat[0]

    x = addList(temp_value, prediction)

    user_graph = px.line(x, x=x['x'], y=x['y'], markers=True)
    user_graph.update_xaxes(type='category')

    user_graph.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Million Square km',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    user_graph.update_yaxes(zeroline=False)

    user_graph.update_traces(hovertemplate="<br>".join([
        "%{x} zettajoules",
        "%{y} km",
        "<extra></extra>"
    ]))

    return fig2, f'The predicted arctic sea ice extent: \
                {prediction} \
                km', user_graph




