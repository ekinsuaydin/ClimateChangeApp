import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc

app = DjangoDash("worldmaps")

countries = json.load(open('GoGreen/dash_apps/finished_apps/map.json', 'r'))

df = pd.read_csv('GoGreen/static/Media/data/countries_co2.csv', dtype={"Code": str})

country_dict_list = [{'label': row['Entity'], 'value': row['Code']} for index, row in
                     df.groupby(['Code', 'Entity']).sum().reset_index().iterrows()]

df2 = pd.read_csv('GoGreen/static/Media/data/countries_fossil_fuel_primary_energy.csv',
                  dtype={"Code": str})

df3 = pd.read_csv('GoGreen/static/Media/data/countries_forest.csv',
                  dtype={"Code": str})

df4 = pd.read_csv('GoGreen/static/Media/data/countries_fossil_fuel_production.csv',
                  dtype={"Code": str})

year_gap = np.arange(start=1950, stop=2021, step=1)

country = df['Entity'].unique()

app.layout = html.Div([
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("Which map would you like view?", style={'color': '#F2F3F4'}),
                dcc.Dropdown(['Carbon Dioxide Map',
                              'Fossil Fuel Consumption Map',
                              'Fossil Fuel Production Map',
                              'Deforestation Map'],
                             value='Carbon Dioxide Map',
                             id='map_type',
                             clearable=False,
                             style={'width': '100%',
                                    'margin-top': '0px'},
                             placeholder='Which map would you like view?'),
            ]),
            html.Br(),
            html.Div([
                html.P('Which year would you like to view on map?', style={'color': '#F2F3F4'}),
                dcc.Dropdown(options=[{'label': y, 'value': y} for y in year_gap],
                             value=2020,
                             id='map_year',
                             clearable=False,
                             style={'width': '100%',
                                    'margin-top': '0px'},
                             placeholder='Which year would you like to view on map?'),
            ]),

        ], width=3),

        dbc.Col([
            dcc.Graph(id='world_map')],
            width=9),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line_graph')],
            width=8),
    ])
], style={'backgroundColor': 'black', })


@app.callback(
    Output('world_map', 'figure'),
    [
        Input('map_year', 'value'),
        Input('map_type', 'value')
    ]
    )
def update_world_map(map_year, map_type):

    if map_type == 'Carbon Dioxide Map':
        currentData = df.loc[df['Year'] == map_year]
        locations = currentData['Code']
        z = currentData['Annual CO2 emissions (per capita)']
        text = currentData['Entity']
        customdata = currentData[['Year', 'Annual CO2 emissions (per capita)']]
        colorscale = "PuBu"

    elif map_type == 'Fossil Fuel Consumption Map':
        currentData = df2.loc[df2['Year'] == map_year]
        locations = currentData['Code']
        z = currentData['Fossil Fuels (TWh)']
        text = currentData['Entity']
        customdata = currentData[['Year', 'Fossil Fuels (TWh)']]
        colorscale = "Oranges"

    elif map_type == 'Fossil Fuel Production Map':
        df4['Total Production'] = df4.iloc[:, -3:].sum(axis=1)
        currentData = df4.loc[df4['Year'] == map_year]
        locations = currentData['Code']
        z = currentData['Total Production']
        text = currentData['Entity']
        customdata = currentData[['Year', 'Total Production']]
        colorscale = "Picnic"

    else:
        currentData = df3[['Code', 'Entity', str(map_year)]]
        locations = currentData['Code']
        z = currentData[str(map_year)]
        text = currentData['Entity']
        currentData['Year'] = str(map_year)
        customdata = currentData[['Year', str(map_year)]]
        colorscale = "greens"

    fig = go.Figure(go.Choroplethmapbox(geojson=countries,
                                        locations=locations,
                                        z=z,
                                        colorscale=colorscale,
                                        showscale=True,
                                        marker_opacity=0.6,
                                        marker_line_width=0.5,
                                        text=text,
                                        customdata=customdata,
                                        hovertemplate=
                                        "<b>%{text}</b><br><br>" +
                                        "%{customdata[0]}<br>" +
                                        "%{customdata[1]}<br>" +
                                        "<extra></extra>",
                                        ))

    fig.update_layout(mapbox_style="carto-darkmatter",
                      paper_bgcolor='black',
                      plot_bgcolor='black',
                      font={"color": "#F2F3F4"},
                      mapbox_center={"lat": 47, "lon": 20},
                      mapbox_zoom=0.5,)
    fig.update_layout(clickmode='event+select')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
    Output('line_graph', 'figure'),
    [
        Input('world_map', 'selectedData'),
        Input('map_type', 'value'),

    ]
)
def update_line_chart(selectedData, map_type):
    country_codes = []
    if not selectedData:
        country_codes = ['TUR']
    else:
        for point in selectedData['points']:
            country_codes.append(point['location'])

    if map_type == 'Carbon Dioxide Map':
        dff = df
        x = 'Year'
        y = 'Annual CO2 emissions (per capita)'
        title = 'CO2 Comparison'
        y_axis_title = 'Annual Co2 Emission Per Capita'

    elif map_type == 'Fossil Fuel Consumption Map':
        dff = df2
        x = 'Year'
        y = 'Fossil Fuels (TWh)'
        title = 'Fossil Fuel Consumption Comparison'
        y_axis_title = 'Fossil Fuels (TWh)'

    elif map_type == 'Fossil Fuel Production Map':
        dff = df4
        x = 'Year'
        y = 'Total Production'
        title = 'Fossil Fuel Production Comparison'
        y_axis_title = 'Fossil Fuels (TWh)'

    else:
        dff = df3
    dff = dff[dff['Code'].isin(country_codes)]

    fig2 = px.line(dff, x=x, y=y,
                   title='Entity',
                   color='Entity',
                   line_group='Entity',
                   # hover_name='Entity'
                   )
    fig2.update_traces(hovertemplate="%{y}")
    fig2.update_xaxes(title='Year', showgrid=True, gridwidth=0.05, gridcolor='gray')
    fig2.update_yaxes(title=y_axis_title, showgrid=True, gridwidth=0.05, gridcolor='gray')
    fig2.update_layout(hovermode="x unified",
                       title=title,
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

    fig2.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

    return fig2
