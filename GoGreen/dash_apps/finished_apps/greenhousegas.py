import dash_html_components as html
from django_plotly_dash import DjangoDash
import pandas as pd
import dash_core_components as dcc
from dash.dependencies import Input, Output
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import plotly.graph_objs as go
import os.path
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("greenhousegas", external_stylesheets=external_stylesheets)

# Data preparation & reading:

# Giving names to co2 dataframe columns
column_names = ['year', 'month', 'decimal', 'average',
                'de-season', 'days', 'st.dev of days', 'unc of mon mean']
# Skip first 53 row comments when reading the data
co2 = pd.read_csv('GoGreen/static/Media/data/co2_mm_mlo.txt',
                  names=column_names, header=0, delimiter='\s+', skiprows=52)


# Giving names to ch4 dataframe columns
column_names2 = ['year', 'month', 'decimal', 'average',
                 'average_unc', 'trend', 'trend_unc']

# Skip first 63 row comments when reading the data
ch4 = pd.read_csv('GoGreen/static/Media/data/ch4_mm_gl.txt',
                  names=column_names2, header=0, delimiter='\s+', skiprows=63)


# Skip first 63 row comments when reading the data
no2 = pd.read_csv('GoGreen/static/Media/data/n2o_mm_gl.txt',
                  names=column_names2, header=0, delimiter='\s+', skiprows=63)


# Fossil Fuel Production Graph
prod_df = pd.read_csv('GoGreen/static/Media/data/countries_fossil_fuel_production.csv',
                  dtype={"Code": str})

prod_df_ = (prod_df[prod_df['Entity'] == 'World']).dropna()

fossil_fuel = go.Figure()

fossil_fuel.add_trace(go.Scatter(
    x=prod_df_['Year'], y=prod_df_['Coal Production - TWh'],
    name='Coal',
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(131, 90, 241)'),
    stackgroup='one'
))

fossil_fuel.add_trace(go.Scatter(
    x=prod_df_['Year'], y=prod_df_['Oil Production - TWh'],
    name='Oil',
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(111, 231, 219)'),
    stackgroup='one'
))
fossil_fuel.add_trace(go.Scatter(
    x=prod_df_['Year'], y=prod_df_['Gas Production - TWh'],
    name='Gas',
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(184, 247, 212)'),
    stackgroup='one'
))

fossil_fuel.update_xaxes(title='Year', showgrid=False)
fossil_fuel.update_yaxes(title='Terawatt-hours', showgrid=True, gridwidth=0.05, gridcolor='gray')
fossil_fuel.update_layout(hovermode="x unified",
                          legend=dict(
                               x=0,
                               y=1.0,
                               title='',
                               bgcolor='black',
                           ),
                          paper_bgcolor='black',
                          plot_bgcolor='black',
                          font={"color": "#D3D3D3"},
                          )


fossil_fuel.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

# Fossil Fuel Production Graph
cons_df = pd.read_csv('GoGreen/static/Media/data/global_fossil_fuel_consumption.csv',
                  dtype={"Code": str})


cons_fig = go.Figure()

cons_fig.add_trace(go.Scatter(
    x=cons_df['Year'], y=cons_df['Coal (TWh; direct energy)'],
    name='Coal',
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(131, 90, 241)'),
    stackgroup='one'
))

cons_fig.add_trace(go.Scatter(
    x=cons_df['Year'], y=cons_df['Oil (TWh; direct energy)'],
    name='Oil',
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(111, 231, 219)'),
    stackgroup='one'
))
cons_fig.add_trace(go.Scatter(
    x=cons_df['Year'], y=cons_df['Gas (TWh; direct energy)'],
    name='Gas',
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(184, 247, 212)'),
    stackgroup='one'
))

cons_fig.update_xaxes(title='Year', showgrid=False)
cons_fig.update_yaxes(title='Terawatt-hours', showgrid=True, gridwidth=0.05, gridcolor='gray')
cons_fig.update_layout(hovermode="x unified",
                          legend=dict(
                               x=0,
                               y=1.0,
                               title='',
                               bgcolor='black',
                           ),
                          paper_bgcolor='black',
                          plot_bgcolor='black',
                          font={"color": "#D3D3D3"},
                          )


cons_fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')


def writecsv(filename, data):
    if os.path.exists(filename):
        return filename
    else:
        with open(f'GoGreen/static/Media/data/{filename}', 'w') as file:
            data.to_csv(file)
        return filename
    return filename


app.layout = html.Div(children=[
    html.Div(children=[
        html.Div(
            html.H6('Global Fossil Fuel Consumption', style={'color': '#F2F3F4'}),
            style={'margin-top': '7px', 'margin-left': '10px'}),
        html.Div(
            dcc.Graph(id="cons_fig", figure=cons_fig),
        ),
    ]),

   html.Div(children=[
            html.Div(
                html.H6('Global Fossil Fuel Production', style={'color': '#F2F3F4'}),
                style={'margin-top': '7px', 'margin-left': '10px'}),
            html.Div(
                dcc.Graph(id="fossil_fuel", figure=fossil_fuel),
            ),
   ]),

   html.Br(),
   html.Div(children=[
        html.Div(children=[
            html.Div(
                html.H6('Carbon Dioxide(CO2) Concentration', style={'color': '#F2F3F4'}),
                style={'margin-left': '10px', 'display': 'inline-block'}),

        ]),
        html.Div(children=[
            dcc.Graph(id='co2_graph'),
            html.Div(
                dcc.Input(id="co2_prediction_year",
                          type="number",
                          min=2021,
                          value=2030),
                style={'margin-left': '10px', 'display': 'inline-block'}),
            html.Div(
                html.P(id="co2_prediction"),
                style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'})
        ])
    ]),
   html.Br(),
   html.Div(children=[
        html.Div(children=[
            html.Div(
                html.H6('Methane(CH4) Concentration', style={'color': '#F2F3F4'}),
                style={'margin-left': '10px', 'display': 'inline-block'}),

        ]),
        html.Div(children=[
            dcc.Graph(id='ch4_graph'),
            html.Div(
                dcc.Input(id="ch4_prediction_year",
                          type="number",
                          min=2021,
                          value=2030),
                style={'margin-left': '10px', 'display': 'inline-block'}),
            html.Div(
                html.P(id="ch4_prediction"),
                style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'})
        ])
    ]),
   html.Br(),
   html.Div(children=[
        html.Div(children=[
            html.Div(
                html.H6('Nitrogen Oxide(N2O) Concentration', style={'color': '#F2F3F4'}),
                style={'margin-left': '10px', 'display': 'inline-block'}),

        ]),
        html.Div(children=[
            dcc.Graph(id='no2_graph'),
            html.Div(
                dcc.Input(id="no2_prediction_year",
                          type="number",
                          min=2021,
                          value=2030),
                style={'margin-left': '10px', 'display': 'inline-block'}),
            html.Div(
                html.P(id="no2_prediction"),
                style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'})
        ])
    ])

])


@app.callback(
    Output("co2_graph", "figure"),
    Output("co2_prediction", "children"),
    [
        Input("co2_prediction_year", "value")
    ]
)
def generate_chart(co2_prediction_year):
    X = np.array(co2['decimal']).reshape(-1, 1)
    y = np.array(co2['average'])
    k1 = 50.0 * RBF(length_scale=50.0)
    k2 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    k3 = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    k4 = 0.1 ** 2 * RBF(length_scale=0.1) + \
         WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, np.inf))

    kernel = k1 + k2 + k3 + k4

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)
    gp.fit(X, y)

    # Prediction
    X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)

    # write csv
    df = co2.groupby('year', as_index=False).last()
    X2 = np.array(df['year']).reshape(-1, 1)
    y2 = np.array(df['average'])
    gp.fit(X2, y2)
    X2_ = np.linspace(X2.min(), X2.max() + 50, 115)[:, np.newaxis]
    y2_pred = gp.predict(X2_)

    data = pd.DataFrame({'X': X2_.reshape(-1),
                         'y': y2_pred,
                         })

    writecsv('co2_prediction', data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(co2['decimal']), y=y,
                             mode="markers", name="Measured",
                             marker=dict(
                                size=5,
                            ),
    ))

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Year',
            showgrid=False,
        ),
        yaxis=dict(
            title='ppm',
            showgrid=True,
            gridcolor='#DCDCDC',
            gridwidth=0.5,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

    fig.update_traces(
        hovertemplate="<br>".join([
                        "%{y} ppm",
                        "<extra></extra>"
                    ]))

    fig.add_trace(go.Scatter(x=X_.reshape(-1), y=y_pred, name='Prediction'))
    fig.add_trace(
        go.Scatter(x=X_[:, 0], y=y_pred + y_std, name='max', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    fig.add_trace(
        go.Scatter(x=X_[:, 0], y=y_pred - y_std, name='min', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))

    prediction = gp.predict([[co2_prediction_year]])
    prediction = prediction.flat[0]

    return fig, f'The predicted CO2: \
                {prediction} \
                ppm'


@app.callback(
    Output("ch4_graph", "figure"),
    Output("ch4_prediction", "children"),
    [
        Input("ch4_prediction_year", "value")
    ]
)
def generate_chart(ch4_prediction_year):
    X = np.array(ch4['decimal']).reshape(-1, 1)
    y = np.array(ch4['average'])
    k1 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    k2 = 0.1 ** 2 * RBF(length_scale=0.1) + \
         WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5))
    k3 = 2.0 ** 2 * RBF(length_scale=1.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')

    kernel = k1 + k2 + k3

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y)

    # Prediction
    X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)

    # write csv
    df = ch4.groupby('year', as_index=False).last()
    X2 = np.array(df['year']).reshape(-1, 1)
    y2 = np.array(df['average'])
    gp.fit(X2, y2)
    X2_ = np.linspace(X2.min(), X2.max() + 50, 90)[:, np.newaxis]
    y2_pred = gp.predict(X2_)

    data = pd.DataFrame({'X': X2_.reshape(-1),
                         'y': y2_pred,
                         })

    writecsv('ch4_prediction', data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(ch4['decimal']), y=y,
                             mode="markers", name="Measured",
                             marker=dict(
                                size=5,
                            ),
    ))

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Year',
            showgrid=False,
        ),
        yaxis=dict(
            title='ppb',
            showgrid=True,
            gridcolor='#DCDCDC',
            gridwidth=0.5,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

    fig.update_traces(hovertemplate="<br>".join([
        "%{y} ppb",
        "<extra></extra>"
    ]))

    fig.add_trace(go.Scatter(x=X_.reshape(-1), y=y_pred, name='Prediction'))
    fig.add_trace(
        go.Scatter(x=X_[:, 0], y=y_pred + y_std, name='max', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    fig.add_trace(
        go.Scatter(x=X_[:, 0], y=y_pred - y_std, name='min', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))

    prediction = gp.predict([[ch4_prediction_year]])
    prediction = prediction.flat[0]

    return fig, f'The predicted CH4 in \
                {ch4_prediction_year} \
                : \
                {prediction} \
                ppb'


@app.callback(
    Output("no2_graph", "figure"),
    Output("no2_prediction", "children"),
    [
        Input("no2_prediction_year", "value"),
    ]
)
def generate_chart(no2_prediction_year):
    X = np.array(no2['decimal']).reshape(-1, 1)
    y = np.array(no2['average'])
    k1 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    k2 = 0.1 ** 2 * RBF(length_scale=0.1) + \
         WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5))
    k3 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')

    kernel = k1 + k2 + k3

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y)

    # Prediction
    X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)

    # write csv
    df = no2.groupby('year', as_index=False).last()
    X2 = np.array(df['year']).reshape(-1, 1)
    y2 = np.array(df['average'])
    gp.fit(X2, y2)
    X2_ = np.linspace(X2.min(), X2.max() + 50, 72)[:, np.newaxis]
    y2_pred = gp.predict(X2_)

    data = pd.DataFrame({'X': X2_.reshape(-1),
                         'y': y2_pred,
                         })

    writecsv('no2_prediction', data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(no2['decimal']), y=y,
                             mode="markers", name="Measured",
                             marker=dict(
                                size=5,
                            ),
    ))

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Year',
            showgrid=False,
        ),
        yaxis=dict(
            title='ppb',
            showgrid=True,
            gridcolor='#DCDCDC',
            gridwidth=0.5,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

    fig.update_traces(hovertemplate="<br>".join([
        "%{y} ppb",
        "<extra></extra>"
    ]))

    fig.add_trace(go.Scatter(x=X_.reshape(-1), y=y_pred, name='Prediction'))
    fig.add_trace(
        go.Scatter(x=X_[:, 0], y=y_pred + y_std, name='max', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    fig.add_trace(
        go.Scatter(x=X_[:, 0], y=y_pred - y_std, name='min', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))

    prediction = gp.predict([[no2_prediction_year]])
    prediction = prediction.flat[0]

    return fig, f'The predicted N20 in \
                {no2_prediction_year} \
                : \
                {prediction} \
                ppb'
