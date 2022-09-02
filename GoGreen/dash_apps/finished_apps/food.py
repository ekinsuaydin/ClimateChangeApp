import dash_core_components as dcc
import dash_html_components as html
from django_plotly_dash import DjangoDash
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("food", external_stylesheets=external_stylesheets)

df = pd.read_csv('GoGreen/static/Media/data/Food_Production.csv')
# First graph: Total Emissions of Food
food_df = df.groupby("Food product", as_index=False)['Total_emissions'].sum()
food_emission_fig = go.Figure(data=go.Scatter(
              x=food_df["Food product"],
              y=food_df["Total_emissions"],
              mode='markers',
              marker=dict(
                  sizemode='diameter',
                  sizeref=1,
                  size=food_df.Total_emissions * 2,
                  color=food_df.Total_emissions,
                  showscale=False
              ))
)
food_emission_fig.update_xaxes(ticklen=5,
                               showgrid=False,
                               zeroline=False,
                               showline=False)
food_emission_fig.update_yaxes(title='Total Emissions',
                               showgrid=False,
                               zeroline=False,
                               ticklen=5,
                               gridwidth=2)
food_emission_fig.update_layout(paper_bgcolor='black',
                                plot_bgcolor='black',
                                hovermode='closest',
                                font={"color": "#D3D3D3"}, )

# Second graph: Greenhouse Gas Emissions Across Supply Chain
supply_emission = df.sort_values(by="Total_emissions", ascending=True).iloc[:, :8]

supply_emission_fig = go.Figure()

supply_emission_fig.add_trace(go.Bar(
        name='Land Usage',
        x=supply_emission['Land use change'],
        y=supply_emission['Food product'],
        orientation='h')
)
supply_emission_fig.add_trace(go.Bar(
        name='Farm',
        x=supply_emission['Farm'],
        orientation='h',
        y=supply_emission['Food product']),
)
supply_emission_fig.add_trace(go.Bar(
        name='Animal Feed',
        x=supply_emission['Animal Feed'],
        y=supply_emission['Food product'],
        orientation='h'),
)

supply_emission_fig.add_trace(go.Bar(
        name='Processing',
        x=supply_emission['Processing'],
        y=supply_emission['Food product'],
        orientation='h'),
)

supply_emission_fig.add_trace(go.Bar(
        name='Transport',
        x=supply_emission['Transport'],
        y=supply_emission['Food product'],
        orientation='h'),
)

supply_emission_fig.add_trace(go.Bar(
        name='Retail',
        x=supply_emission['Retail'],
        y=supply_emission['Food product'],
        orientation='h'),
)

supply_emission_fig.add_trace(go.Bar(
        name='Packaging',
        x=supply_emission['Packging'],
        y=supply_emission['Food product'],
        orientation='h'),
)
supply_emission_fig.update_layout(barmode='stack', height=900)

supply_emission_fig.update_xaxes(showgrid=False,
                                 zeroline=False,
                                 showline=False)

supply_emission_fig.update_yaxes(showgrid=False,
                                 zeroline=False,)

supply_emission_fig.update_layout(paper_bgcolor='black',
                                  plot_bgcolor='black',
                                  hovermode='closest',
                                  font={"color": "#D3D3D3"},
                                  )

# Third Graph: Greenhouse Emissions

greenhouse_emissions_df = df.dropna().sort_values(by='Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)',
                                                  ascending=True)[['Food product', 'Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)']]

greenhouse_emissions_fig = px.bar(greenhouse_emissions_df,
                                  x="Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)",
                                  y="Food product",
                                  orientation='h'
                                  )

greenhouse_emissions_fig.update_layout(height=750)

greenhouse_emissions_fig.update_xaxes(showgrid=False,
                                      zeroline=False,
                                      showline=False)

greenhouse_emissions_fig.update_yaxes(title='',
                                      showgrid=False,
                                      zeroline=False,)

greenhouse_emissions_fig.update_layout(paper_bgcolor='black',
                                       plot_bgcolor='black',
                                       font={"color": "#D3D3D3"},
                                       )
greenhouse_emissions_fig.update_layout(hovermode="y unified")
greenhouse_emissions_fig.update_traces(hovertemplate="<br>".join([
        "%{x}",
        "<extra></extra>"
    ])
)

# Fourth Graph: Land Use
land_use_df = df.dropna().sort_values(by='Land use per 1000kcal (m² per 1000kcal)',
                                      ascending=True)[['Food product', 'Land use per 1000kcal (m² per 1000kcal)']]


land_use_fig = px.bar(land_use_df,
                      x="Land use per 1000kcal (m² per 1000kcal)",
                      y="Food product",
                      orientation='h'
                      )


land_use_fig.update_layout(height=750)

land_use_fig.update_xaxes(showgrid=False,
                          zeroline=False,
                          showline=False)


land_use_fig.update_yaxes(title='',
                          showgrid=False,
                          zeroline=False, )

land_use_fig.update_layout(paper_bgcolor='black',
                           plot_bgcolor='black',
                           font={"color": "#D3D3D3"},
                           )
land_use_fig.update_layout(hovermode="y unified")
land_use_fig.update_traces(marker_color='#EF553B',
                           hovertemplate="<br>".join([
                               "%{x}",
                               "<extra></extra>"
                           ]))

# Fifth Graph: Land Use
water_use_df = df.dropna().sort_values(by='Freshwater withdrawals per 1000kcal (liters per 1000kcal)',
                                       ascending=True)[['Food product','Freshwater withdrawals per 1000kcal (liters per 1000kcal)']]


water_use_fig = px.bar(water_use_df,
                       x="Freshwater withdrawals per 1000kcal (liters per 1000kcal)",
                       y="Food product",
                       orientation='h'
                       )


water_use_fig.update_layout(height=750)

water_use_fig.update_xaxes(showgrid=False,
                           zeroline=False,
                           showline=False)

water_use_fig.update_yaxes(title='',
                           showgrid=False,
                           zeroline=False, )

water_use_fig.update_layout(paper_bgcolor='black',
                            plot_bgcolor='black',
                            font={"color": "#D3D3D3"},
                            )
water_use_fig.update_layout(hovermode="y unified")
water_use_fig.update_traces(marker_color='#00CC96',
                            hovertemplate="<br>".join([
                               "%{x}",
                               "<extra></extra>"
                            ]))

# Sixth Graph: Land Use
eutrophication_df = df.dropna().sort_values(by='Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)',
                                            ascending= True)[['Food product', 'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)']]


eutrophication_fig = px.bar(eutrophication_df,
                            x="Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)",
                            y="Food product",
                            orientation='h')

eutrophication_fig.update_layout(height=750)

eutrophication_fig.update_xaxes(showgrid=False,
                                zeroline=False,
                                showline=False)

eutrophication_fig.update_yaxes(title='',
                                showgrid=False,
                                zeroline=False, )

eutrophication_fig.update_layout(paper_bgcolor='black',
                                 plot_bgcolor='black',
                                 font={"color": "#D3D3D3"},
                                 )
eutrophication_fig.update_layout(hovermode="y unified")
eutrophication_fig.update_traces(marker_color='#FFA5BA',
                                 hovertemplate="<br>".join([
                                   "%{x}",
                                   "<extra></extra>"
                                 ]))

app.layout = html.Div(children=[
    html.Div(
            html.H6('Total Emissions of Foods', style={'color': '#F2F3F4'}),
            style={'margin-left': '10px', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(id='food_emissions', figure=food_emission_fig),
    ),

    html.Br(),

    html.Div(
            html.H6('Greenhouse Gas Emissions Across the Supply Chain', style={'color': '#F2F3F4'}),
            style={'margin-left': '10px', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(id='supply_emissions', figure=supply_emission_fig),
    ),

    html.Br(),

    html.Div(
            html.H6('Greenhouse Gas Emissions per 1000 kcal', style={'color': '#F2F3F4'}),
            style={'margin-left': '10px', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(id='greenhouse_emissions', figure=greenhouse_emissions_fig),
    ),

    html.Br(),

    html.Div(
            html.H6('Land Usage per 1000 kcal', style={'color': '#F2F3F4'}),
            style={'margin-left': '10px', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(id='land_usage', figure=land_use_fig),
    ),

    html.Br(),

    html.Div(
            html.H6('Water Usage per 1000 kcal', style={'color': '#F2F3F4'}),
            style={'margin-left': '10px', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(id='water_usage', figure=water_use_fig),
    ),

    html.Br(),

    html.Div(
            html.H6('Eutrophication Emissions per 1000 kcal', style={'color': '#F2F3F4'}),
            style={'margin-left': '10px', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(id='eutrophying', figure=eutrophication_fig),
    ),

    html.Br(),

    html.Div([
        html.H6('Do you want to calculate correlations?', style={'color': '#F2F3F4'}),
        html.P('Choose one of each', style={'color': '#F2F3F4'}),

    ],  style={'margin-left': '10px'}
    ),

    html.Div([

        html.Div(
            dcc.Dropdown(
              id='corr_first',
              options=[
                  {'label': 'Greenhouse Emissions of Foods',
                   'value': 'Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)'},
                  {'label': 'Eutrophication of Foods',
                   'value': 'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)'},
                  {'label': 'Water Usage of Foods',
                   'value': 'Freshwater withdrawals per 1000kcal (liters per 1000kcal)'},
                  {'label': 'Land Usage of Foods',
                   'value': 'Land use per 1000kcal (m² per 1000kcal)'},
              ],
              style={'width': '100%'},
            ),
            style={'width': '30%', 'display': 'inline-block'}),

        html.Div(
            dcc.Dropdown(
              id='corr_second',
              options=[
                  {'label': 'Greenhouse Emissions of Foods',
                   'value': 'Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)'},
                  {'label': 'Eutrophication of Foods',
                   'value': 'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)'},
                  {'label': 'Water Usage of Foods',
                   'value': 'Freshwater withdrawals per 1000kcal (liters per 1000kcal)'},
                  {'label': 'Land Usage of Foods',
                   'value': 'Land use per 1000kcal (m² per 1000kcal)'},
              ],
              style={'width': '100%'},

            ),
            style={'margin-left': '20px', 'width': '30%', 'display': 'inline-block'}),

    ], style={'margin-left': '10px'}),

    html.Div(
        html.P(id="correlation"),
        style={'margin-left': '10px', 'display': 'inline-block', 'color': '#F2F3F4'})
])


@app.callback(
    Output("correlation", "children"),
    [
        Input("corr_first", "value"),
        Input("corr_second", "value")
    ]
)
def correlation(corr_first, corr_second):
    corr = df[corr_first].corr(df[corr_second])
    if corr>0.8:
        correlation = 'Too High!'
    elif 0.8 > corr > 0.6:
        correlation = 'High!'
    elif 0.6 > corr > 0.4:
        correlation = 'Low.'
    else:
        correlation = 'Very Low.'

    return 'Correlation between them is ', correlation