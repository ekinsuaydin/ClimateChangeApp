import numpy as np
from django.shortcuts import render, redirect
from .models import Image
from .models import UserUpload
import pandas as pd
from plotly.offline import plot
from PIL import Image as im
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
import plotly.graph_objs as go
import calendar
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from django.http import HttpResponse, JsonResponse

user_data = {"x": [], "y": []}
df1 = pd.DataFrame(user_data)
user_data2 = {"x": [], "y": []}
user_data3 = {"x": [], "y": []}


def addList1(data):
    df1 = pd.DataFrame(data)
    return df1

def addList2(x, y):
    user_data2["x"].append(x)
    user_data2["y"].append(y)
    df2 = pd.DataFrame(user_data2)
    return df2


def addList3(x, y):
    user_data3["x"].append(x)
    user_data3["y"].append(y)
    df3 = pd.DataFrame(user_data3)
    return df3


def checkchange(value):
    if value > 0:
        value = '+' + str(value) + '%'
    else:
        value = str(value) + '%'
    return value


def home(request):
    # Data preparation & reading:

    # Giving names to co2 dataframe columns
    column_names1 = ['Year No', 'Smoothing', 'Lowess(5)']

    # Skip first 5 row comments when reading the data
    temp_df = pd.read_csv('GoGreen/static/Media/data/temperature.txt',
                          names=column_names1, header=0, delimiter='\s+', skiprows=4)

    temp_change = ((100 * (temp_df['Smoothing'].iloc[-1] - temp_df['Smoothing'].iloc[0]))
                   / temp_df['Smoothing'].iloc[-1])

    temp_change = checkchange(int(temp_change))

    # Giving names to co2 dataframe columns
    column_names2 = ['year', 'month', 'decimal', 'average',
                     'de-season', 'days', 'st.dev of days', 'unc of mon mean']
    # Skip first 53 row comments when reading the data
    co2 = pd.read_csv('GoGreen/static/Media/data/co2_mm_mlo.txt',
                      names=column_names2, header=0, delimiter='\s+', skiprows=52)

    co2_change = ((100 * (co2['average'].iloc[-1] - co2['average'].iloc[0]))
                  / co2['average'].iloc[-1])

    co2_change = checkchange(int(co2_change))
    co2['month'] = co2['month'].apply(lambda x: calendar.month_abbr[x])
    co2['year'] = pd.to_datetime(co2['year'].astype(str) + '/' + co2['month'].astype(str) + '/01')

    fig_co2 = go.Figure(go.Scatter(x=co2['year'], y=co2['average']))
    fig_co2.update_layout(xaxis=dict(
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

    fig_co2.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_co2.update_layout(hovermode="x unified")
    fig_co2.update_traces(hovertemplate="<br>".join([
        "%{y} ppm",
        "<extra></extra>"
    ])
    )
    plot_co2 = plot(fig_co2, output_type='div', include_plotlyjs=False)

    sea_ice = pd.read_excel('GoGreen/static/Media/data/2485_Sept_Arctic_extent_1979-2021.xlsx', engine='openpyxl')
    sea_ice_change = ((100 * (sea_ice['extent'].iloc[-1] - sea_ice['extent'].iloc[0]))
                      / sea_ice['extent'].iloc[-1])
    sea_ice_change = checkchange(int(sea_ice_change))

    fig_sea_ice = go.Figure(go.Scatter(x=sea_ice['year'], y=sea_ice['extent']))

    fig_sea_ice.update_layout(xaxis=dict(
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

    fig_sea_ice.update_layout(hovermode="x unified")

    fig_sea_ice.update_traces(hovertemplate="<br>".join([
        "%{y} million square km",
        "<extra></extra>"
    ])
    )
    plot_sea_ice = plot(fig_sea_ice, output_type='div', include_plotlyjs=False)
    column_names = ['type', '#', 'decimal', 'num of obs',
                    'obs1', 'obs2', 'obs3', 'obs4',
                    'obs5', 'obs6', 'obs7', 'obs8']

    # Skip first 47 row comments when reading the data
    sea_level = pd.read_csv('GoGreen/static/Media/data/GMSL_TPJAOS_5.1_199209_202203.txt',
                            names=column_names, header=0, delimiter='\s+', skiprows=47)
    sea_level_change = ((100 * (sea_level['obs8'].iloc[-1] - sea_level['obs8'].iloc[0]))
                        / sea_level['obs8'].iloc[-1])
    sea_level_change = checkchange(int(sea_level_change))
    fig_sea_level = px.line(sea_level, x=sea_level['decimal'], y=sea_level['obs8'])

    fig_sea_level.update_layout(xaxis=dict(
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

    fig_sea_level.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_sea_level.update_layout(hovermode="x unified")
    fig_sea_level.update_traces(hovertemplate="<br>".join([
        "%{y} mm",
        "<extra></extra>"
    ])
    )
    plot_sea_level = plot(fig_sea_level, output_type='div', include_plotlyjs=False)


    context = {
        'temp_change': temp_change,
        'co2_change': co2_change,
        'sea_ice_change': sea_ice_change,
        'sea_level_change': sea_level_change,
        'fig_co2': plot_co2,
        'fig_sea_ice': plot_sea_ice,
        'fig_sea_level': plot_sea_level

    }

    return render(request, 'GoGreen/home.html', context)


def worldmaps(request):
    return render(request, 'GoGreen/worldmaps.html', {})


def footprintcalculator(request):
    return render(request, 'GoGreen/footprintcalculator.html', {})


def imageslider(request):
    return render(request, 'GoGreen/imageslider.html', {})


def causesbase(request):
    return render(request, 'GoGreen/causesbase.html', {})


def effectsbase(request):
    return render(request, 'GoGreen/effectsbase.html', {})


def causes(request):
    return render(request, 'GoGreen/causes.html', {})


def effects(request):
    return render(request, 'GoGreen/effects.html', {})


def temperature(request):
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
    temp = pd.read_csv('GoGreen/static/Media/data/temperature.txt',
                     names=column_names, header=0, delimiter='\s+', skiprows=5)
    co2_temp = co2_.merge(temp, how='inner', left_on='year', right_on='Year No')
    fig_co2 = px.line(temp, x=temp['Year No'], y=temp['Smoothing'])
    fig_co2.update_xaxes(title='Year', showgrid=False, gridwidth=0.05, gridcolor='gray')
    fig_co2.update_yaxes(title='Temperature Anomaly(C°)', showgrid=True, gridwidth=0.05, gridcolor='gray')
    fig_co2.update_layout(
        hovermode="x unified",
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#D3D3D3"}, )
    fig_co2.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_co2.update_traces(hovertemplate="<br>".join([
        "%{y} C°",
        "<extra></extra>"
    ]))
    plot_fig_co2 = plot(fig_co2, output_type='div', include_plotlyjs=False)

    X = temp.iloc[:, 0:1]
    y = temp.iloc[:, 1]

    X_ = np.linspace(X.min(), X.max() + 51, 192)[:, np.newaxis]
    X_ = X_.reshape((X_.shape[0], -1), order='F')

    polynomial_regression = PolynomialFeatures(degree=2)
    x_poly = polynomial_regression.fit_transform(X)
    polynomial_regression.fit(x_poly, y)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    temperature_pred = lin_reg.predict(polynomial_regression.fit_transform(X_))

    if request.method == 'POST' and request.POST['action'] == 'first_prediction':
        prediction_year = int(request.POST.get('prediction_year'))
        print(prediction_year)
        prediction = lin_reg.predict(polynomial_regression.fit_transform([[prediction_year]]))
        prediction = prediction.flat[0]
        print(prediction)

        data = {'prediction_year': prediction_year, 'prediction':prediction}
        return JsonResponse(data)

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

    X2 = np.array(co2_temp['average'])
    y2 = np.array(co2_temp['Smoothing'])
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    poly_features2 = poly2.fit_transform(X2.reshape(-1, 1))
    poly_reg_model2 = LinearRegression()
    poly_reg_model2.fit(poly_features2, y2)
    y_predicted = poly_reg_model2.predict(poly_features2)

    if request.method == 'POST' and request.POST['action'] == 'second_prediction':
        prediction_co2 = int(request.POST.get('co2'))
        print(prediction_co2)
        prediction2 = poly_reg_model2.predict(poly2.fit_transform([[prediction_co2]]))
        prediction2 = prediction2.flat[0]
        print(prediction2)
        data = {'co2': prediction_co2, 'prediction': prediction2}
        user_data["x"].append(prediction_co2)
        user_data["y"].append(prediction2)
        df2 = pd.DataFrame(user_data)
        addList1(df2)
        print(df1)
        return JsonResponse(data)

    user_graph = px.line(df1, x='x', y='y', markers=True)
    user_graph.update_xaxes(type='category')

    user_graph.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Carbon Dioxide(CO2) ppm',
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

    plot_user_graph = plot(user_graph, output_type='div', include_plotlyjs=False)

    plot_co2temp = plot(co2_temp_fig, output_type='div', include_plotlyjs=False)

    context = {
        'plot_fig_co2': plot_fig_co2,
        'plot_co2_temp': plot_co2temp,
        'plot_user_graph': plot_user_graph
    }

    return render(request, 'GoGreen/temperature.html', context)


def sealevel(request):

    # Giving names to co2 dataframe columns
    column_names = ['Year No', 'Smoothing', 'Lowess(5)']

    # Giving names to dataframe columns
    column_names = ['type', '#', 'decimal', 'num of obs',
                    'obs1', 'obs2', 'obs3', 'obs4',
                    'obs5', 'obs6', 'obs7', 'obs8']

    # Skip first 47 row comments when reading the data
    df = pd.read_csv('GoGreen/static/Media/data/GMSL_TPJAOS_5.1_199209_202203.txt',
                     names=column_names, header=0, delimiter='\s+', skiprows=47)

    fig_sealevel = px.line(df, x=df['decimal'], y=df['obs8'])
    fig_sealevel.update_xaxes(title='Year', showgrid=False)
    fig_sealevel.update_yaxes(title='Sea Level Change(mm)', showgrid=False, gridwidth=0.05, gridcolor='gray')
    fig_sealevel.update_layout(paper_bgcolor='black',
                               plot_bgcolor='black',
                               font={"color": "#D3D3D3"}, )

    fig_sealevel.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_sealevel.update_layout(hovermode="x unified")
    fig_sealevel.update_traces(hovertemplate="<br>".join([
        "%{y} mm",
        "<extra></extra>"
    ])
    )

    # Regression
    X = np.array(df['decimal'])
    y = np.array(df['obs8'])
    poly = PolynomialFeatures(degree=1, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    if request.method == 'POST' and 'prediction1' in request.POST:
        prediction_year = int(request.POST['prediction_year'])
        print(prediction_year)
        prediction = poly_reg_model.predict(poly.fit_transform([[prediction_year]]))
        prediction = prediction.flat[0]
        print(prediction)
    else:
        prediction_year = 2030
        prediction = poly_reg_model.predict(poly.fit_transform([[prediction_year]]))
        prediction = prediction.flat[0]

    plot_sealevel = plot(fig_sealevel, output_type='div', include_plotlyjs=False)

    # Ocean Heat Temperature Relation

    # Giving names to co2 dataframe columns
    temp_column_names = ['Year No', 'Smoothing', 'Lowess(5)']
    # Skip first 5 row comments when reading the data
    temp = pd.read_csv('GoGreen/static/Media/data/temperature.txt',
                       names=temp_column_names, header=0, delimiter='\s+', skiprows=5)

    ocean_heat = pd.read_csv('GoGreen/static/Media/data/pent_h22-w0-2000m.dat.txt', header=0, delimiter='\s+')
    ocean_heat['Year'] = (ocean_heat['YEAR'].astype(str).str[:4]).astype(int)

    oceanheat_temp = ocean_heat.merge(temp, how='inner', left_on='Year', right_on='Year No')

    fig_oceanheat_temp = px.scatter(oceanheat_temp, x=oceanheat_temp['Smoothing'],
                                    y=oceanheat_temp['NH'])
    fig_oceanheat_temp.update_yaxes(title='Ocean Heat(zettajoules)', showgrid=False)
    fig_oceanheat_temp.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Temperature Anomaly(C°)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )
    fig_oceanheat_temp.update_xaxes(zeroline=False)
    fig_oceanheat_temp.update_yaxes(zeroline=False)
    fig_oceanheat_temp.update_traces(hovertemplate="<br>".join([
        "%{x} C°",
        "%{y} zettajoules",
        "<extra></extra>"
    ])
    )

    X2 = np.array(oceanheat_temp['Smoothing'])
    y2 = np.array(oceanheat_temp['NH'])

    poly2 = PolynomialFeatures(degree=1, include_bias=False)
    poly_features2 = poly2.fit_transform(X2.reshape(-1, 1))
    poly_reg_model2 = LinearRegression()
    poly_reg_model2.fit(poly_features2, y2)
    y_predicted = poly_reg_model2.predict(poly_features2)

    if request.method == 'POST' and 'prediction2' in request.POST:
        prediction_year2 = float(request.POST['prediction_year2'])
        print(prediction_year2)
        prediction2 = poly_reg_model2.predict(poly2.fit_transform([[prediction_year2]]))
        prediction2 = prediction2.flat[0]
        print(prediction2)
    else:
        prediction_year2 = None
        prediction2 = None

    x = addList2(prediction_year2, prediction2)

    user_graph = px.line(x, x=x['x'], y=x['y'], markers=True)
    user_graph.update_xaxes(type='category')

    user_graph.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Temperature Anomaly(C°)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    user_graph.update_yaxes(zeroline=False)

    user_graph.update_traces(hovertemplate="<br>".join([
        "%{x} C°",
        "%{y} zettajoules",
        "<extra></extra>"
    ]))

    plot_user_graph = plot(user_graph, output_type='div', include_plotlyjs=False)

    plot_oceanheat_temp = plot(fig_oceanheat_temp, output_type='div', include_plotlyjs=False)

    context = {
        'plot_sealevel': plot_sealevel,
        'plot_oceanheat_temp': plot_oceanheat_temp,
        'prediction_year': prediction_year,
        'prediction': prediction,
        'prediction_year2': prediction_year2,
        'prediction2': prediction2,
        'plot_user_graph': plot_user_graph
    }

    return render(request, 'GoGreen/sealevel.html', context)


def arcticsea(request):
    arctic_sea = pd.read_excel('GoGreen/static/Media/data/2485_Sept_Arctic_extent_1979-2021.xlsx', engine='openpyxl')
    ocean_heat = pd.read_csv('GoGreen/static/Media/data/pent_h22-w0-2000m.dat.txt', header=0, delimiter='\s+')
    ocean_heat['Year'] = (ocean_heat['YEAR'].astype(str).str[:4]).astype(int)
    oceanheat_seaice = ocean_heat.merge(arctic_sea, how='inner', left_on='Year', right_on='year')
    
    fig_arcticsea = px.line(arctic_sea, x=arctic_sea['year'], y=arctic_sea['extent'])
    fig_arcticsea.update_xaxes(title='Year', showgrid=False)
    fig_arcticsea.update_yaxes(title='Million Square km', showgrid=True, gridwidth=0.05, gridcolor='gray')
    fig_arcticsea.update_layout(paper_bgcolor='black',
                      plot_bgcolor='black',
                      font={"color": "#D3D3D3"}, )

    fig_arcticsea.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_arcticsea.update_layout(hovermode="x unified")
    fig_arcticsea.update_traces(hovertemplate="<br>".join([
        "%{y} million square km",
        "<extra></extra>"
    ]))

    X = np.array(arctic_sea['year'])
    y = np.array(arctic_sea['extent'])
    # Create model for the class to fit in
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    if request.method == 'POST' and 'prediction1' in request.POST:
        prediction_year = int(request.POST['prediction_year'])
        print(prediction_year)
        prediction = poly_reg_model.predict(poly.fit_transform([[prediction_year]]))
        prediction = prediction.flat[0]
        print(prediction)
    else:
        prediction_year = 2030
        prediction = poly_reg_model.predict(poly.fit_transform([[prediction_year]]))
        prediction = prediction.flat[0]

    plot_arcticsea = plot(fig_arcticsea, output_type='div', include_plotlyjs=False)

    fig_ocean_seaice = px.scatter(oceanheat_seaice, x=oceanheat_seaice['NH'], y=oceanheat_seaice['extent'])
    fig_ocean_seaice.update_yaxes(title='Ocean Heat(zettajoules)', showgrid=False)
    fig_ocean_seaice.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Million Square km',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    fig_ocean_seaice.update_xaxes(zeroline=False)
    fig_ocean_seaice.update_yaxes(zeroline=False)
    fig_ocean_seaice.update_traces(hovertemplate="<br>".join([
        "%{x} zettajoules",
        "%{y} million square km",
        "<extra></extra>"
    ])
    )

    plot_oceanheat_seaice = plot(fig_ocean_seaice, output_type='div', include_plotlyjs=False)

    X2 = np.array(oceanheat_seaice['NH'])
    y2 = np.array(oceanheat_seaice['extent'])

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    poly_features2 = poly2.fit_transform(X2.reshape(-1, 1))
    poly_reg_model2 = LinearRegression()
    poly_reg_model2.fit(poly_features2, y2)
    y_predicted = poly_reg_model2.predict(poly_features2)

    if request.method == 'POST' and 'prediction2' in request.POST:
        prediction_year2 = float(request.POST['prediction_year2'])
        print(prediction_year2)
        prediction2 = poly_reg_model2.predict(poly2.fit_transform([[prediction_year2]]))
        prediction2 = prediction2.flat[0]
        print(prediction2)
    else:
        prediction_year2 = None
        prediction2 = None

    x = addList3(prediction_year2, prediction2)

    user_graph = px.line(x, x=x['x'], y=x['y'], markers=True)
    user_graph.update_xaxes(type='category')

    user_graph.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title='Ocean Heat(zettajoules)',
            showgrid=False,
        ),
        yaxis=dict(
            title='Million Square km',
            showgrid=False,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={"color": "#DCDCDC"}, )

    user_graph.update_yaxes(zeroline=False)

    user_graph.update_traces(hovertemplate="<br>".join([
        "%{x} zettajoules",
        "%{y} km",
        "<extra></extra>"
    ]))

    plot_user_graph = plot(user_graph, output_type='div', include_plotlyjs=False)

    context = {
        'plot_arcticsea': plot_arcticsea,
        'prediction_year': prediction_year,
        'prediction': prediction,
        'plot_oceanheat_seaice': plot_oceanheat_seaice,
        'prediction_year2': prediction_year2,
        'prediction2': prediction2,
        'plot_user_graph': plot_user_graph


    }
    return render(request, 'GoGreen/iceSheets.html', context)


def greenhousegas(request):
    # CO2
    # Data preparation & reading:

    # Giving names to co2 dataframe columns
    column_names_co2 = ['year', 'month', 'decimal', 'average',
                    'de-season', 'days', 'st.dev of days', 'unc of mon mean']
    # Skip first 53 row comments when reading the data
    co2 = pd.read_csv('GoGreen/static/Media/data/co2_mm_mlo.txt',
                      names=column_names_co2, header=0, delimiter='\s+', skiprows=52)

    X_co2 = np.array(co2['decimal']).reshape(-1, 1)
    y_co2 = np.array(co2['average'])

    # kernel calculations are from sklearn
    k1_co2 = 50.0 * RBF(length_scale=50.0)
    k2_co2 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    k3_co2 = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    k4_co2 = 0.1 ** 2 * RBF(length_scale=0.1) + \
         WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, np.inf))

    kernel = k1_co2 + k2_co2 + k3_co2 + k4_co2

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)
    gp.fit(X_co2, y_co2)

    # Prediction
    X_co2_ = np.linspace(X_co2.min(), X_co2.max() + 30, 1000)[:, np.newaxis]
    y_pred_co2, y_std_co2 = gp.predict(X_co2_, return_std=True)
    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(x=np.array(co2['decimal']), y=y_co2,
                             mode="markers", name="Measured",
                             marker=dict(
                                 size=5,
                             ),
                             ))
    fig_co2.update_layout(
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
    fig_co2.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_co2.update_traces(
        hovertemplate="<br>".join([
            "%{y} ppm",
            "<extra></extra>"
        ]))
    fig_co2.add_trace(go.Scatter(x=X_co2_.reshape(-1), y=y_pred_co2, name='Prediction'))
    fig_co2.add_trace(
        go.Scatter(x=X_co2_[:, 0], y=y_pred_co2 + y_std_co2, name='max', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    fig_co2.add_trace(
        go.Scatter(x=X_co2_[:, 0], y=y_pred_co2 - y_std_co2, name='min', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    plot_co2 = plot(fig_co2, output_type='div', include_plotlyjs=False)

    # ch4
    # Giving names to ch4 dataframe columns
    column_names2 = ['year', 'month', 'decimal', 'average',
                     'average_unc', 'trend', 'trend_unc']

    # Skip first 63 row comments when reading the data
    ch4 = pd.read_csv('GoGreen/static/Media/data/ch4_mm_gl.txt',
                      names=column_names2, header=0, delimiter='\s+', skiprows=63)

    X_ch4 = np.array(ch4['decimal']).reshape(-1, 1)
    y_ch4 = np.array(ch4['average'])
    k1_ch4 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    k2_ch4 = 0.1 ** 2 * RBF(length_scale=0.1) + \
         WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5))
    k3_ch4 = 2.0 ** 2 * RBF(length_scale=1.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')

    kernel = k1_ch4 + k2_ch4 + k3_ch4

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X_ch4, y_ch4)

    # Prediction
    X_ch4_ = np.linspace(X_ch4.min(), X_ch4.max() + 30, 1000)[:, np.newaxis]
    y_pred_ch4, y_std_ch4 = gp.predict(X_ch4_, return_std=True)
    fig_ch4 = go.Figure()
    fig_ch4.add_trace(go.Scatter(x=np.array(ch4['decimal']), y=y_ch4,
                             mode="markers", name="Measured",
                             marker=dict(
                                 size=5,
                             ),
                             ))
    fig_ch4.update_layout(
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
    fig_ch4.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')
    fig_ch4.update_traces(hovertemplate="<br>".join([
        "%{y} ppb",
        "<extra></extra>"
    ]))
    fig_ch4.add_trace(go.Scatter(x=X_ch4_.reshape(-1), y=y_pred_ch4, name='Prediction'))
    fig_ch4.add_trace(
        go.Scatter(x=X_ch4_[:, 0], y=y_pred_ch4 + y_std_ch4, name='max', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    fig_ch4.add_trace(
        go.Scatter(x=X_ch4_[:, 0], y=y_pred_ch4 - y_std_ch4, name='min', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    plot_ch4 = plot(fig_ch4, output_type='div', include_plotlyjs=False)
    # Skip first 63 row comments when reading the data
    no2 = pd.read_csv('GoGreen/static/Media/data/n2o_mm_gl.txt',
                      names=column_names2, header=0, delimiter='\s+', skiprows=63)

    X_no2 = np.array(no2['decimal']).reshape(-1, 1)
    y_no2 = np.array(no2['average'])
    k1_no2 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    k2_no2 = 0.1 ** 2 * RBF(length_scale=0.1) + \
         WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5))
    k3_no2 = 2.0 ** 2 * RBF(length_scale=100.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')

    kernel = k1_no2 + k2_no2 + k3_no2

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X_no2, y_no2)

    # Prediction
    X_no2_ = np.linspace(X_no2.min(), X_no2.max() + 30, 1000)[:, np.newaxis]
    y_pred_no2, y_std_no2 = gp.predict(X_no2_, return_std=True)

    fig_no2 = go.Figure()
    fig_no2.add_trace(go.Scatter(x=np.array(no2['decimal']), y=y_no2,
                             mode="markers", name="Measured",
                             marker=dict(
                                 size=5,
                             ),
                             ))

    fig_no2.update_layout(
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

    fig_no2.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='#DCDCDC')

    fig_no2.update_traces(hovertemplate="<br>".join([
        "%{y} ppb",
        "<extra></extra>"
    ]))

    fig_no2.add_trace(go.Scatter(x=X_no2_.reshape(-1), y=y_pred_no2, name='Prediction'))
    fig_no2.add_trace(
        go.Scatter(x=X_no2_[:, 0], y=y_pred_no2 + y_std_no2, name='max', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    fig_no2.add_trace(
        go.Scatter(x=X_no2_[:, 0], y=y_pred_no2 - y_std_no2, name='min', line=dict(color="#808080", width=0.5),
                   fill='tonexty'))
    plot_no2 = plot(fig_no2, output_type='div', include_plotlyjs=False)

    context = {
        'plot_co2': plot_co2,
        'plot_ch4': plot_ch4,
        'plot_no2': plot_no2,
    }

    return render(request, 'GoGreen/greenhousegas.html', context)


def deforestation(request):
    images = Image.objects.all()

    if request.method == 'POST':
        data = request.POST
        image = request.FILES.get('image')

        print('data', data)
        print('image', image)

        uploaded_image = UserUpload.objects.create(
            date=data['date'],
            location=data['location'],
            area=data['area'],
            cluster=data['cluster'],
            image=image
        )

        return redirect('deforesttationanalyze', uploaded_image.id)

    context = {
        'images': images
    }

    return render(request, 'GoGreen/deforestation.html', context)


def deforestationanalyze(request, pk):
    cluster = int(request.GET.get('cluster'))
    images = Image.objects.all()
    image = Image.objects.get(id=pk)
    url = "GoGreen/static" + image.image.url
    image_ = im.open(url)
    width, height = image_.size

    I1 = image_.convert('L')
    I2 = np.asarray(I1, dtype=np.float)
    Y = I2.reshape((-1, 1))
    k_means_ = KMeans(n_clusters=cluster)
    k_means_.fit(Y)
    centroids_ = k_means_.cluster_centers_
    labels_ = k_means_.labels_
    I2compressed = np.choose(labels_, centroids_)
    I2compressed.shape = I2.shape
    fig = px.imshow(image_)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        width=width * 0.2,
        height=height * 0.2,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        hovermode=False
    )

    fig2 = px.imshow(I2compressed, color_continuous_scale='gray')

    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    fig2.update_layout(
        width=width * 0.2,
        height=height * 0.2,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        hovermode=False
    )

    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)

    # Get colors
    I2 = (I2compressed - np.min(I2compressed)) / (np.max(I2compressed) - np.min(I2compressed)) * 255
    I2 = im.fromarray(I2.astype(np.uint8))
    colors = I2.getcolors(width * height)
    colors2 = [item[0] for item in colors]

    if cluster == 2:
        names = ["Black", "White"]
        colors_ = ["#000000", "#ffffff"]

    elif cluster == 3:
        names = ["Black", "Gray", "White"]
        colors_ = ["#000000", "#333333", "#ffffff"]

    elif cluster == 4:
        names = ["Black", "Gray", "Light Gray", "White"]
        colors_ = ["#000000", "#333333", "#666666", "#ffffff"]

    pie_chart = px.pie(values=colors2, names=names, height=320)
    pie_chart.update_layout(paper_bgcolor='black',
                            plot_bgcolor='black',
                            font={"color": "#F2F3F4"},
                            )
    pie_chart.update_traces(hovertemplate="%{label}: <br> %{value} number of pixels",
                            marker=dict(colors=colors_, line=dict(color='#ffffff', width=1)))
    pie_chart.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    plot_div3 = plot(pie_chart, output_type='div', include_plotlyjs=False)

    context = {
        'images': images,
        'image': image,
        'plot': plot_div,
        'plot2': plot_div2,
        'cluster': cluster,
        'plot3': plot_div3
    }

    return render(request, 'GoGreen/deforestationanalyze.html', context)


def deforestationanalyze2(request, pk):
    image = UserUpload.objects.get(id=pk)
    cluster = image.cluster
    area = image.area
    location = image.location
    date = image.date

    url = "GoGreen/static" + image.image.url
    image_ = im.open(url)
    width, height = image_.size

    I1 = image_.convert('L')
    I2 = np.asarray(I1, dtype=np.float)
    Y = I2.reshape((-1, 1))
    k_means_ = KMeans(n_clusters=cluster)
    k_means_.fit(Y)
    centroids_ = k_means_.cluster_centers_
    labels_ = k_means_.labels_
    I2compressed = np.choose(labels_, centroids_)
    I2compressed.shape = I2.shape
    fig = px.imshow(image_)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        width=width * 0.2,
        height=height * 0.2,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        hovermode=False
    )

    fig2 = px.imshow(I2compressed, color_continuous_scale='gray')

    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    fig2.update_layout(
        width=width * 0.2,
        height=height * 0.2,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        hovermode=False
    )

    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)

    # Get colors
    I2 = (I2compressed - np.min(I2compressed)) / (np.max(I2compressed) - np.min(I2compressed)) * 255
    I2 = im.fromarray(I2.astype(np.uint8))
    colors = I2.getcolors(width * height)
    colors2 = [item[0] for item in colors]
    if area != '':
        print(area)
        newList = [item*area for item in colors2]
        print(newList)
        total_pixels = sum(colors2)
        print(total_pixels)
        newList2 = [item/total_pixels for item in newList]
        print(newList2)
        colors2 = newList2
    else:
        pass

    if cluster == 2:
        names = ["Black", "White"]
        colors_ = ["#000000", "#ffffff"]

    elif cluster == 3:
        names = ["Black", "Gray", "White"]
        colors_ = ["#000000", "#333333", "#ffffff"]

    elif cluster == 4:
        names = ["Black", "Gray", "Light Gray", "White"]
        colors_ = ["#000000", "#333333", "#666666", "#ffffff"]

    pie_chart = px.pie(values=colors2, names=names, height=320)
    pie_chart.update_layout(paper_bgcolor='black',
                            plot_bgcolor='black',
                            font={"color": "#F2F3F4"},
                            )
    pie_chart.update_traces(hovertemplate="%{label}: <br> %{value} km2",
                            marker=dict(colors=colors_, line=dict(color='#ffffff', width=1)))
    pie_chart.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    plot_div3 = plot(pie_chart, output_type='div', include_plotlyjs=False)

    context = {
        'image': image,
        'cluster': cluster,
        'area': area,
        'location': location,
        'date': date,
        'plot': plot_div,
        'plot2': plot_div2,
        'plot3': plot_div3
    }

    return render(request, 'GoGreen/deforestationanalyze2.html', context)


def food(request):
    return render(request, 'GoGreen/food.html', {})


def forestfire(request):
    return render(request, 'GoGreen/forestfire.html', {})


