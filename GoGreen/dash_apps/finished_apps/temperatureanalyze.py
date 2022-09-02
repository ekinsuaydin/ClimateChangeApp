import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from django_plotly_dash import DjangoDash
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objs as go
import os.path
import os
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("temperatureanalyze", external_stylesheets=external_stylesheets)

# Giving names to co2 dataframe columns
column_names = ['year', 'month', 'decimal', 'average',
                'de-season', 'days', 'st.dev of days', 'unc of mon mean']
# Skip first 53 row comments when reading the data
co2 = pd.read_csv('GoGreen/static/Media/data/co2_mm_mlo.txt',
                  names=column_names, header=0, delimiter='\s+', skiprows=52)
co2_ = co2.groupby('year', as_index=False).last()

# Data preparation & reading:

# Giving names to co2 dataframe columns
column_names = ['Year No', 'Smoothing', 'Lowess(5)']

# Skip first 5 row comments when reading the data
df = pd.read_csv('GoGreen/static/Media/data/temperature.txt',
                 names=column_names, header=0, delimiter='\s+', skiprows=5)

co2_temp = co2_.merge(df, how='inner', left_on='year', right_on='Year No')

user_data = {"x": [], "y": []}
df2 = pd.DataFrame(user_data)


def addList(x, y):
    user_data["x"].append(x)
    user_data["y"].append(y)
    df2 = pd.DataFrame(user_data)
    return df2


def writecsv(filename, data):
    if os.path.exists(filename):
        return filename
    else:
        with open(f'GoGreen/static/Media/data/{filename}', 'w') as file:
            data.to_csv(file)
        return filename
    return filename


app.layout = html.Div(children=[
    html.H6('Temperature Change from 1880 to 2021', style={'color': '#F2F3F4'}),

    html.Div(children=[
        html.Div(
            dcc.Dropdown(
                        ['Celsius', 'Fahrenheit'], 'Celsius',
                        id='degree',
                        clearable=False,
                        style={'width': '100%'},
            ), style={'width': '25%', 'display': 'inline-block'}
        ),
        html.Div(
            daq.BooleanSwitch(id="prediction_line", on=False, color="#515A5A"),
            style={'float': 'right', 'display': 'inline-block'}),
    ]),

    dcc.Graph(id='temperature_graph'),
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
    html.Div([
        html.Div(
            html.H6('Carbon-dioxide Increases, Temperature Increases!', style={'color': '#F2F3F4'}),
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div(
            daq.BooleanSwitch(id="prediction_line2", on=False, color="#515A5A"),
            style={'float': 'right', 'display': 'inline-block'}
        ),

    ]),
    dcc.Graph(id='co2_temp_graph'),
    html.Br(),
    html.P('Type(ppm) to predict temperature anomaly!', style={'color': '#F2F3F4'}),
    html.Div(
       dcc.Input(id="prediction_2",
                 type="number",
                 min=200,
                 value=500),
       style={'display': 'inline-block'}),
    html.Div(
        html.P(id="prediction2"),
        style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'}),
    html.Br(),
    html.H5("CO2 vs Annual Temperature Graph of Your Predictions", style={'color': '#F2F3F4'}),
    dcc.Graph(id='user_graph'),

], style={'margin-left': '10px'})


@app.callback(
    Output("temperature_graph", "figure"),
    Output("prediction", "children"),
    [
        Input("degree", "value"),
        Input("prediction_year", "value"),
        Input("prediction_line", "on")
    ]
)
def generate_chart(degree, prediction_year, on):
    if degree == 'Celsius':
        # defining y for polynomial regression
        y = df.iloc[:, 1]
        fig = px.line(df, x=df['Year No'], y=df['Smoothing'])
        fig.update_xaxes(title='Year', showgrid=True, gridwidth=0.05, gridcolor='gray')
        fig.update_yaxes(title='Temperature Anomaly(C°)', showgrid=True, gridwidth=0.05, gridcolor='gray')
        fig.update_layout(
                          hovermode="x unified",
                          paper_bgcolor='black',
                          plot_bgcolor='black',
                          font={"color": "#D3D3D3"}, )

        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

        fig.update_traces(hovertemplate="<br>".join([
            "%{y} C°",
            "<extra></extra>"
        ]))

    else:
        df['Fahrenheit'] = df['Smoothing']*0.8
        # defining y for polynomial regression
        y = df.iloc[:, 3]
        fig = px.line(df, x=df['Year No'], y=df['Fahrenheit'])
        fig.update_xaxes(title='Year', showgrid=True, gridwidth=0.05, gridcolor='gray')
        fig.update_yaxes(title='Temperature Anomaly(°F)', showgrid=True, gridwidth=0.05, gridcolor='gray')
        fig.update_layout(
                          hovermode="x unified",
                          paper_bgcolor='black',
                          plot_bgcolor='black',
                          font={"color": "#D3D3D3"}, )

        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
        fig.update_traces(hovertemplate="<br>".join([
            "%{x}",
            "%{y} °F",
            "<extra></extra>"
        ]))

    # Polynomial Regression
    X = df.iloc[:, 0:1]
    # Adding new 172 axis
    X_ = np.linspace(X.min(), X.max() + 51, 192)[:, np.newaxis]

    X_ = X_.reshape((X_.shape[0], -1), order='F')
    polynomial_regression = PolynomialFeatures(degree=2)
    x_poly = polynomial_regression.fit_transform(X)
    polynomial_regression.fit(x_poly, y)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    temperature_pred = lin_reg.predict(polynomial_regression.fit_transform(X_))

    data = pd.DataFrame({'X': X_.reshape(-1),
                         'y': temperature_pred,
                         })

    writecsv('temperature_prediction', data)

    switch = "{}".format(on)
    if switch == "True":
        fig.add_shape(
            type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
            x0=0, x1=1, xref="paper", y0=1.5, y1=1.5, yref="y"
        )

        fig.add_trace(go.Scatter(
            x=X_.reshape(-1),
            y=temperature_pred,
            line_color='#00FF7F',
            name='Prediction',
            opacity=0.8)
        )

    prediction = lin_reg.predict(polynomial_regression.fit_transform([[prediction_year]]))
    prediction = prediction.flat[0]

    return fig,  f'The predicted temperature: \
                {prediction} \
                C°'


@app.callback(
    Output("co2_temp_graph", "figure"),
    Output("prediction2", "children"),
    Output("user_graph", "figure"),
    [
        Input("prediction_2", "value"),
        Input("prediction_line2", "on")
    ]
)
def generate_chart(prediction_2, on):
    co2_temp_fig = px.scatter(co2_temp, x=co2_temp['average'], y=co2_temp['Smoothing'])

    co2_temp_fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='CO2(ppm)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Temperature Anomaly(C°)',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    co2_temp_fig.update_yaxes(zeroline=False)

    co2_temp_fig.update_traces(hovertemplate="<br>".join([
        "%{x} ppm",
        "%{y} C°",
        "<extra></extra>"
    ]))

    X = np.array(co2_temp['average'])
    y = np.array(co2_temp['Smoothing'])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    switch = "{}".format(on)
    if switch == "True":
        co2_temp_fig.add_trace(go.Scatter(
            x=X.reshape(-1),
            y=y_predicted,
            line_color='#00FF7F',
            name='Prediction',
            opacity=0.8)
        )

    prediction = poly_reg_model.predict(poly.fit_transform([[prediction_2]]))
    prediction = prediction.flat[0]

    x = addList(prediction_2, prediction)

    user_graph = px.line(x, x=x['x'], y=x['y'], markers=True)
    user_graph.update_xaxes(type='category')

    user_graph.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='CO2(ppm)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Temperature Anomaly(C°)',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    user_graph.update_yaxes(zeroline=False)

    user_graph.update_traces(hovertemplate="<br>".join([
        "%{x} ppm",
        "%{y} C°",
        "<extra></extra>"
    ]))

    return co2_temp_fig, f'The predicted temperature: \
                            {prediction} \
                            C°', user_graph
