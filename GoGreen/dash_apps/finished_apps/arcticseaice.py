import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("arcticseaice", external_stylesheets=external_stylesheets)


df = pd.read_excel('GoGreen/static/Media/data/2485_Sept_Arctic_extent_1979-2021.xlsx', engine='openpyxl')

fig = go.Figure(go.Scatter(x=df['year'], y=df['extent']))

fig.update_layout(xaxis=dict(
    title='Year',
    showgrid=False,
),
    yaxis=dict(
        title='Million Square km',
        showgrid=True,
        gridcolor='#DCDCDC',
        gridwidth=1,
    ),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF')

fig.update_layout(hovermode="x unified")

fig.update_traces(hovertemplate="<br>".join([
        "%{y} million square km",
        "<extra></extra>"
    ])
)

# fig.update_traces(line=dict(color='#00BFFF'))

app.layout = html.Div([
    html.Div([dcc.Graph(id='sea_ice_graph', figure=fig)])])


