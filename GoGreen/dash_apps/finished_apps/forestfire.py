import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
from django_plotly_dash import DjangoDash
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("forestfire", external_stylesheets=external_stylesheets)

df = pd.read_csv('GoGreen/static/Media/data/forestfires.csv')

fig = px.scatter(df, x="temp", y="area")


fig.update_xaxes(showgrid=False,
                 zeroline=False,
                 showline=False)

fig.update_yaxes(title='Correlations',
                 showgrid=False,
                 zeroline=False,)

fig.update_layout(paper_bgcolor='black',
                  plot_bgcolor='black',
                  hovermode='closest',
                  font={"color": "#D3D3D3"}, )

app.layout = html.Div([
    html.Div(
        dcc.Graph(id='corr', figure=fig),
        style={'margin-left': '10px'}
    ),
])


