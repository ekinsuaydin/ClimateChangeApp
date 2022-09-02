import dash_core_components as dcc
import dash_html_components as html
from django_plotly_dash import DjangoDash
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("temperature", external_stylesheets=external_stylesheets)

# Data preparation & reading:

# Giving names to co2 dataframe columns
column_names = ['Year No', 'Smoothing', 'Lowess(5)']

# Skip first 5 row comments when reading the data
df = pd.read_csv('GoGreen/static/Media/data/temperature.txt',
                 names=column_names, header=0, delimiter='\s+', skiprows=5)


app.layout = html.Div(children=[
    html.P("Choose one of them to see the changes in degrees Celsius or Fahrenheit"),
    html.Div(
        dcc.Dropdown(
            ['Celsius', 'Fahrenheit'], 'Celsius',
            id='degree',
            clearable=False,
            style={'width': '100%'},
        ), style={'width': '25%', 'display': 'inline-block'}
    ),
    dcc.Graph(id='temperature_graph')
])


@app.callback(
    Output("temperature_graph", "figure"),
    Input("degree", "value"))
def generate_chart(degrees):
    if degrees == 'Celsius':
        fig = px.line(df, x=df['Year No'], y=df['Smoothing'])
        fig.update_layout(xaxis=dict(
                          title='Year',
                          showgrid=False,
                          ),
                          yaxis=dict(
                            title='C°',
                            showgrid=True,
                            gridcolor='#DCDCDC',
                            gridwidth=1,
                          ),
                          paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#FFFFFF')

        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
        fig.update_layout(hovermode="x unified")

        fig.update_traces(hovertemplate="<br>".join([
            "%{y} C°",
            "<extra></extra>"
        ]))
    else:
        df['Smoothing'] = df['Smoothing']*0.8
        fig = px.line(df, x=df['Year No'], y=df['Smoothing'])
        fig.update_layout(xaxis=dict(
            title='Year',
            showgrid=False,
        ),
            yaxis=dict(
                title='°F',
                showgrid=True,
                gridcolor='#DCDCDC',
                gridwidth=1,
            ),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF')

        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
        fig.update_layout(hovermode="x unified")
        fig.update_traces(hovertemplate="<br>".join([
            "%{y} °F",
            "<extra></extra>"
        ]))

    return fig





