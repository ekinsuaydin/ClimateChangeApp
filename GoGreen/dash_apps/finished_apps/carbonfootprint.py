import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from django_plotly_dash import DjangoDash
from dash.dependencies import Input, Output
import plotly.express as px

app = DjangoDash("carbonfootprint")

app.layout = html.Div(
    [
        html.Br(),
        html.H4("Calculate Your Carbon Footprint", style={'color': '#F2F3F4',
                                                          'textAlign': 'center'}),
        html.Div([

            html.P("Your monthly electric bill($)", style={'color': '#F2F3F4'}),
            dbc.Input(
                type="number",
                id="electricity",
                min=0,
                max=200,
                value=50,
                step=1,
                style={'width': '100%'},
            ),
            html.P("Your monthly gas bill($)", style={'color': '#F2F3F4'}),
            dbc.Input(
                type="number",
                id="gas",
                min=0,
                max=200,
                value=50,
                step=1,
                style={'width': '100%'},
            ),
            html.P("Your monthly oil bill($)", style={'color': '#F2F3F4'}),
            dbc.Input(
                type="number",
                id="oil",
                min=0,
                max=200,
                value=50,
                step=1,
                style={'width': '100%'},

            ),
            html.P("Your total yearly mileage on your car(km)", style={'color': '#F2F3F4'}),
            dbc.Input(
                type="number",
                id="mileage",
                min=0,
                max=60000,
                value=1000,
                step=1,
                style={'width': '100%'},
            ),
            html.P("Number of flights you’ve taken in the past year", style={'color': '#F2F3F4'}),
            dbc.Input(
                id="flight",
                type="number",
                value=1,
                style={'width': '100%'},
            ),
            html.P("Do you recycle paper?", style={'color': '#F2F3F4'}),
            dcc.Dropdown(
                ['Yes', 'No'], 'Yes',
                id='paper_recycle',
                clearable=False,
                style={'width': '100%'},

            ),
            html.P("Are you a smoker?", style={'color': '#F2F3F4'}),
            dcc.Dropdown(
                ['Yes', 'No'], 'Yes',
                id='smoker',
                clearable=False,
                style={'width': '100%'},

            ),
            html.P("Do you recycle aluminum and tin?", style={'color': '#F2F3F4'}),
            dcc.Dropdown(
                ['Yes', 'No'], 'Yes',
                id='aluminium_recycle',
                clearable=False,
                style={'width': '100%', },
            ), ], style={'display': 'inline-block',
                         'width': '30%',
                         'margin-left': '200px'}),
        html.Div(children=[
            dcc.Graph(id="pie_chart"),
            html.P(id="footprint",
                    style={"textAlign": "left", 'color': '#F2F3F4'},),
        ], style={'display': 'inline-block',
                  'width': '30%',
                  'margin-top': '0px',
                  'margin-left': '30px'
                  }
        )

], style={'background-color': 'black', 'align': 'center'})


@app.callback(

        Output("pie_chart", "figure"),
        Output("footprint", "children"),
    [
        Input("electricity", "value"),
        Input("gas", "value"),
        Input("oil", "value"),
        Input("mileage", "value"),
        Input("flight", "value"),
        Input("paper_recycle", "value"),
        Input("aluminium_recycle", "value"),
        Input("smoker", "value"),

    ],

)
def generate_chart(electricity,
                   gas, oil, mileage,
                   flight, paper_recycle,
                   aluminium_recycle, smoker):
    if int(flight) < 4:
        flight = flight*1100
    else:
        flight = flight*4400

    if paper_recycle == 'Yes':
        x = 0
    else:
        x = 184

    if aluminium_recycle == 'Yes':
        y = 0
    else:
        y = 166

    if smoker == 'Yes':
        z = 200
    else:
        z = 0



    transport = (float(mileage) * 0.79) + flight
    energy = ((float(electricity) * 105) + (float(gas) * 105))
    consumption = (float(oil) * 113) + float(x) + float(y) + z

    footprint = int((transport+energy+consumption)/1000)

    if footprint > 22:
        message = 'Shame on you! You may want to take some “living green” practices into consideration.'
    elif 22 > footprint > 6:
        message = 'You are in the average.'
    else:
        message = 'Wow! Applaud by shouting.'

    names = ['Transport', 'Energy', 'Consumption']
    values = [transport, energy, consumption]

    df = px.data.tips()

    fig = px.pie(df, values=values, names=names, hole=.3, height=320)
    fig.update_layout(paper_bgcolor='black',
                      plot_bgcolor='black',
                      font={"color": "#F2F3F4"},
                      )
    fig.update_traces(hovertemplate="%{label}: <br> %{value} pounds per year")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig, ('Your footprint is ', footprint, ' tonnes per year. ', message)
