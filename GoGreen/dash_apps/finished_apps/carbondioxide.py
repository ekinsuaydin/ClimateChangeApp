import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
import pandas as pd
import calendar

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash("carbondioxide", external_stylesheets=external_stylesheets)

# Data preparation & reading:

# Giving names to co2 dataframe columns
column_names = ['year', 'month', 'decimal', 'average',
                'de-season', 'days', 'st.dev of days', 'unc of mon mean']
# Skip first 53 row comments when reading the data
df = pd.read_csv('GoGreen/static/Media/data/co2_mm_mlo.txt',
                  names=column_names, header=0, delimiter='\s+', skiprows=53)


# cells that has the value, '-99.99' are unmeasured. So we have to remove those rows.
# index_names = df[df['fit'] == -99.99].index
# df.drop(index_names, inplace=True)

# convert int months with month names
df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])
df['year'] = pd.to_datetime(df['year'].astype(str) + '/' + df['month'].astype(str) + '/01')

fig = go.Figure(go.Scatter(x=df['year'], y=df['average']))

fig.update_layout(xaxis=dict(
    title='Year',
    showgrid=False,
),
    yaxis=dict(
        title='Co2(ppm)',
        showgrid=True,
        gridcolor='#DCDCDC',
        gridwidth=1,
    ),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF')

fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
fig.update_layout(hovermode="x unified")
fig.update_traces(hovertemplate="<br>".join([
        "%{y} ppm",
        "<extra></extra>"
    ])
)


app.layout = html.Div([
    html.Div([dcc.Graph(id='graph2', figure=fig)])])


