import dash_core_components as dcc
import dash_html_components as html
from django_plotly_dash import DjangoDash
import pandas as pd
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("sealevelrise", external_stylesheets=external_stylesheets)

# Data preparation & reading:

# Giving names to dataframe columns
column_names = ['type', '#', 'decimal', 'num of obs',
                'obs1', 'obs2', 'obs3', 'obs4',
                'obs5', 'obs6', 'obs7', 'obs8']

# Skip first 47 row comments when reading the data
df = pd.read_csv('GoGreen/static/Media/data/GMSL_TPJAOS_5.1_199209_202203.txt',
                 names=column_names, header=0, delimiter='\s+', skiprows=47)

# 1. Get first 4 char from the column year and put it into new column named Year:
# df['Year'] = df['decimal'].astype(str).str[:4]
# 2. Grouping the dataframe by year and get the last row from each group:
# df2 = df.groupby('Year', as_index=False).last()
fig = px.line(df, x=df['decimal'], y=df['obs8'])


fig.update_layout(xaxis=dict(
    title='Year',
    showgrid=False,
),
    yaxis=dict(
        title='Sea Level Change(mm)',
        showgrid=True,
        gridcolor='#DCDCDC',
        gridwidth=1,
    ),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF')

fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
fig.update_layout(hovermode="x unified")
fig.update_traces(hovertemplate="<br>".join([
        "%{y} mm",
        "<extra></extra>"
    ])
)


app.layout = html.Div([
    html.Div([dcc.Graph(id='graph', figure=fig)])
])



