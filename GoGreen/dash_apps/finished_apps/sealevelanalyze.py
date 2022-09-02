import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from django_plotly_dash import DjangoDash
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("sealevelanalyze", external_stylesheets=external_stylesheets)

# Giving names to co2 dataframe columns
column_names = ['Year No', 'Smoothing', 'Lowess(5)']

# Skip first 5 row comments when reading the data
temp = pd.read_csv('GoGreen/static/Media/data/temperature.txt',
                 names=column_names, header=0, delimiter='\s+', skiprows=5)

# Giving names to dataframe columns
column_names = ['type', '#', 'decimal', 'num of obs',
                'obs1', 'obs2', 'obs3', 'obs4',
                'obs5', 'obs6', 'obs7', 'obs8']

# Skip first 47 row comments when reading the data
df = pd.read_csv('GoGreen/static/Media/data/GMSL_TPJAOS_5.1_199209_202203.txt',
                 names=column_names, header=0, delimiter='\s+', skiprows=47)

ocean_heat = pd.read_csv('GoGreen/static/Media/data/pent_h22-w0-2000m.dat.txt', header=0, delimiter='\s+')
ocean_heat['Year'] = (ocean_heat['YEAR'].astype(str).str[:4]).astype(int)

oceanheat_temp = ocean_heat.merge(temp, how='inner', left_on='Year', right_on='Year No')

user_data = {"x": [], "y": []}
df2 = pd.DataFrame(user_data)


def addList(x, y):
    user_data["x"].append(x)
    user_data["y"].append(y)
    df2 = pd.DataFrame(user_data)
    return df2


app.layout = html.Div(children=[
    html.H5("Sea Level Rise", style={'color': '#F2F3F4'}),
    html.Div(
        daq.BooleanSwitch(id="prediction_line", on=False, color="#515A5A"),
        style={'display': 'inline-block'}),

    dcc.Graph(id='sealevel_graph'),
    html.Br(),
    html.P('Which year would you like to predict?', style={'color': '#F2F3F4'}),
    html.Div(
        dcc.Input(id="prediction_year",
                  type="number",
                  min=2021,
                  value=2030),
        style={'display': 'inline-block'}),
    html.Div(
        html.P(id="prediction"),
        style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'}),
    html.Br(),
    html.H5("Temperature Rises, The Ocean Heat Rises!", style={'color': '#F2F3F4'}),
    html.Div(
        daq.BooleanSwitch(id="prediction_line2", on=False, color="#515A5A"),
        style={'display': 'inline-block'}),

    dcc.Graph(id='oceanheat_temp'),
    html.Br(),
    html.P('Enter temperature value', style={'color': '#F2F3F4'}),
    html.Div(
        dcc.Input(id="temp_value",
                  type="number",
                  min=0,
                  value=1.2),
        style={'display': 'inline-block'}),
    html.Div(
        html.P(id="prediction2"),
        style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'}),
    html.Br(),
    html.H5("Temperature vs Ocean Heat Graph of Your Predictions", style={'color': '#F2F3F4'}),
    dcc.Graph(id='user_graph'),

], style={'margin-left': '10px'})


@app.callback(
    Output("sealevel_graph", "figure"),
    Output("prediction", "children"),
    [
        Input("prediction_year", "value"),
        Input("prediction_line", "on")
    ]
)
def generate_chart(prediction_year, on):
    fig = px.line(df, x=df['decimal'], y=df['obs8'])
    fig.update_xaxes(title='Year', showgrid=False)
    fig.update_yaxes(title='Sea Level Change(mm)', showgrid=True, gridwidth=0.05, gridcolor='gray')
    fig.update_layout(paper_bgcolor='black',
                      plot_bgcolor='black',
                      font={"color": "#D3D3D3"}, )

    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig.update_layout(hovermode="x unified")
    fig.update_traces(hovertemplate="<br>".join([
        "%{y} mm",
        "<extra></extra>"
    ])
    )

    X = np.array(df['decimal'])
    y = np.array(df['obs8'])
    # Create model for the class to fit in
    poly = PolynomialFeatures(degree=1, include_bias=False)
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

    return fig, f'The predicted sea level is: \
                            {prediction} \
                            mm',


@app.callback(
    Output("oceanheat_temp", "figure"),
    Output("prediction2", "children"),
    Output("user_graph", "figure"),
    [
        Input("temp_value", "value"),
        Input("prediction_line2", "on")
    ]
)
def generate_chart(temp_value, on):
    fig2 = px.scatter(oceanheat_temp, x=oceanheat_temp['Smoothing'], y=oceanheat_temp['NH'])
    fig2.update_yaxes(title='Ocean Heat(zettajoules)', showgrid=False)
    fig2.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Temperature Anomaly(C°)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    fig2.update_xaxes(zeroline=False)
    fig2.update_yaxes(zeroline=False)
    fig2.update_traces(hovertemplate="<br>".join([
        "%{x}",
        "%{y}",
        "<extra></extra>"
    ])
    )

    X = np.array(oceanheat_temp['Smoothing'])
    y = np.array(oceanheat_temp['NH'])

    poly = PolynomialFeatures(degree=1, include_bias=False)
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
            title='Temperature Anomaly(C°)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    user_graph.update_yaxes(zeroline=False)

    user_graph.update_traces(hovertemplate="<br>".join([
        "%{x} C°",
        "%{y} zettajoules",
        "<extra></extra>"
    ]))

    return fig2, f'The predicted ocean heat is: \
                            {prediction} \
                            zettajoules', user_graph
