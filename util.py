import pandas as pd
import numpy as np     # used for data analysis functionalities
from io import BytesIO
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

# from sklearn.datasets import fetch_olivetti_faces
import plotly.graph_objs as go    # for setting graphs, plots, charts
import matplotlib.pyplot as plt    # matplotlib is imported for plotting
from scipy import special                  # since we're using Bessel functions (bassel functions are special functions)
import seaborn as sns             # for visualizing 
import scipy.special as spl
import plotly.figure_factory as ff
import plotly.graph_objects as go
import urllib, json
import json
import streamlit as st
# Import all required functions and classes from Plotly
import plotly.express as px


import plotly.graph_objs as go
from plotly.figure_factory import create_table
import random

def render_bessel_2():
    x = np.linspace(0, np.pi, 500)   #Creating an X variable, with 500 points

    #labels below. Layout = Title, yaxis is y-axis label, xaxis is x-axis label, but xaxis and yaxis have to be in dictionary
    # so we use dict() with each: the xaxis and yaxis

    # we create our 1st trace, 

    layout = go.Layout(
        title='<b>BESSEL FUNCTIONS PLOT</b>',
        yaxis=dict(
            title='<i>(in Power units)</i>'
        ),
        xaxis=dict(
            title='<i>(in time units)</i>'
        )
    )


    def change_plot(signals, freq):    #Define function with two parameters
        
        """
        This will modify the plot whenever widget changes
        """

        data = []    #empty list called, data to place our traces in 
        for signal in signals:                 # signals is items in multi-select widget
            trace1 = go.Scatter(
                x=x,
                y=spl.jv(signal, freq * x),        #scipy.special.jv(v, z) = <ufunc 'jv'>   #j is the symbol of the bessel function
                mode='lines',                      #no markers since there are 500 points, would be too cluttered
                name='bessel line {}'.format(signal),
                line=dict(
                    shape='spline'   # hv = staircase-ish lines, sorta-pixelated
                )
            )
            data.append(trace1)

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    import ipywidgets as widgets  
    #3 widget's section below 
    signals = widgets.SelectMultiple(options=list(range(8)), value=(0, ), description='Bessel No.')
    freq = widgets.FloatSlider(min=1, max=20, value=1, description='Frequency')
    widgets.interactive(change_plot, signals=signals, freq=freq)

def render_bessel():
    plt.clf()
    x = np.linspace(0, np.pi, 500)   #Creating an X variable, with 500 points

    layout = go.Layout(
        title='<b>BESSEL FUNCTIONS PLOT</b>',
        yaxis=dict(
            title='<i>(in Power units)</i>'
        ),
        xaxis=dict(
            title='<i>(in time units)</i>'
        )
    )

    def change_plot(signals, freq):    # Define function with two parameters
        
        """
        This will modify the plot whenever widget changes
        """

        data = []    #empty list called, data to place our traces in 
        for signal in signals:                 # signals is items in multi-select widget
            trace1 = go.Scatter(
                x=x,
                y=spl.jv(signal, freq * x),        #scipy.special.jv(v, z) = <ufunc 'jv'>   #j is the symbol of the bessel function
                mode='lines',                      #no markers since there are 500 points, would be too cluttered
                name='bessel line {}'.format(signal),
                line=dict(
                    shape='spline'   # hv = staircase-ish lines, sorta-pixelated
                )
            )
            data.append(trace1)

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
    st.write('Select Bessel Number:')
    signals = []
    one = st.checkbox('1')
    two = st.checkbox('2')
    three = st.checkbox('3')
    four = st.checkbox('4')
    five = st.checkbox('5')
    six = st.checkbox('6')
    seven = st.checkbox('7')
    if one:
        signals.append(1)
    if two:
        signals.append(2)
    if three:
        signals.append(3)
    if four:
        signals.append(4)
    if five:
        signals.append(5)
    if six:
        signals.append(6)
    if seven:
        signals.append(7)
    freq = st.slider(min_value=1.0, max_value=20.0, step=0.1, value=1.0, label='Frequency:')
    change_plot(signals, freq)

def render_face_images_scikit():
    plt.clf()
    fig,ax=plt.subplots(5,5, figsize=(5,5))
    fig.subplots_adjust(hspace=0,wspace=0)

    # Through scikitlearn, we are importing face-data - sounds crude

    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces().images

    for i in range(5):
        for j in range(5):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(faces[10 * i + j], cmap="bone")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

def render_markers():
    plt.clf()
    rng = np.random.RandomState(0)
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(rng.rand(5), rng.rand(5), marker,label="marker='{0}'".format(marker))

    plt.legend(numpoints=1)
    plt.xlim(0, 1.8);

    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_contour_plot():
    plt.clf()
    # Visualizing a Three-Dimensional Function
    # function formula used is z = f (x, y)

    def f(x,y):
        return np.sin(x)**10+np.cos(10+y*x)*np.cos(x)
    

    x=np.linspace(0,5,60)
    y=np.linspace(0,5,50)

    # np.meshgrid function, generates a 2-D grids from 1-D arrays

    X,Y=np.meshgrid(x,y)
    Z=f(X,Y)

    #plotting contour plot below 
    plt.contour(X,Y,Z,color='black')
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)
 
def render_hand_written_numbers():
    plt.clf()
    from sklearn.datasets import load_digits
    digits=load_digits(n_class=9)

    fig,ax=plt.subplots(8,8,figsize=(6, 6))
    for i, axi in enumerate(ax.flat):
        axi.imshow(digits.images[i], cmap='binary')
        axi.set(xticks=[], yticks=[])
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_sankey_diagram():
    plt.clf()
    url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    # override gray link colors with 'source' colors
    opacity = 0.4
    # change 'magenta' to its 'rgba' value to add opacity
    data['data'][0]['node']['color'] = ['rgba(255,0,255, 0.8)' if color == "magenta" else color for color in data['data'][0]['node']['color']]
    data['data'][0]['link']['color'] = [data['data'][0]['node']['color'][src].replace("0.8", str(opacity))
                                        for src in data['data'][0]['link']['source']]

    fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        valuesuffix = "TWh",
        # Define nodes
        node = dict(
        pad = 15,
        thickness = 15,
        line = dict(color = "black", width = 0.5),
        label =  data['data'][0]['node']['label'],
        color =  data['data'][0]['node']['color']
        ),
        # Add links
        link = dict(
        source =  data['data'][0]['link']['source'],
        target =  data['data'][0]['link']['target'],
        value =  data['data'][0]['link']['value'],
        label =  data['data'][0]['link']['label'],
        color =  data['data'][0]['link']['color']
    ))])

    fig.update_layout(title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",
                    font_size=10)
    st.plotly_chart(fig, use_container_width=True)

def render_parallel_categories_plot():
    plt.clf()
    # parallel_categories
    df_tips = pd.read_csv("tips.csv")
    fig = px.parallel_categories(df_tips, dimensions=['sex', 'smoker', 'day'],
                    color="size", color_continuous_scale=px.colors.sequential.Sunset,
                    labels={'sex':'Gender of Customer', 'smoker':'Customer is  Smoker', 'day':'Weekday'})
    st.plotly_chart(fig, use_container_width=True)

def render_stock_data_line_chart():
    plt.clf()
    df_stocks = px.data.stocks()
    px.line(df_stocks, x='date', y='GOOG', labels={'x':'Date', 'y':'Price'})


    px.line(df_stocks, x='date', y=['GOOG','AAPL'], labels={'x':'Date', 'y':'Price'},
        title='Apple Vs. Google')


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.AAPL, 
                            mode='lines', name='Apple'))
    fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.AMZN, 
                            mode='lines+markers', name='Amazon'))

    fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.GOOG, 
                            mode='lines+markers', name='Google',
                            line=dict(color='firebrick', width=2, dash='dashdot')))


    fig.update_layout(
        
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_stock_data_box_plot_scatter_plot():
    plt.clf()
    df_tips = px.data.tips()

    px.box(df_tips, x='sex', y='tip', points='all')


    px.box(df_tips, x='day', y='tip', color='sex')


    fig = go.Figure()
    fig.add_trace(go.Box(x=df_tips.sex, y=df_tips.tip, marker_color='blue',
                        boxmean='sd'))


    df_stocks = px.data.stocks()
    fig = go.Figure()

    fig.add_trace(go.Box(y=df_stocks.GOOG, boxpoints='all', name='Google',
                        fillcolor='blue', jitter=0.5, whiskerwidth=0.2))
    fig.add_trace(go.Box(y=df_stocks.AAPL, boxpoints='all', name='Apple',
                        fillcolor='red', jitter=0.5, whiskerwidth=0.2))

    fig.update_layout(title='Google vs. Apple', 
                    yaxis=dict(gridcolor='rgb(255, 255, 255)',
                    gridwidth=3),
                    paper_bgcolor='rgb(243, 243, 243)',
                    plot_bgcolor='rgb(243, 243, 243)')
    st.plotly_chart(fig, use_container_width=True)

def render_gap_minder_table():
    gapminder = px.data.gapminder()
    st.table(gapminder.head(15))

def render_gap_minder_bar_plot():
    plt.clf()
    gapminder = px.data.gapminder()  #Don't comment out please
    fig = px.bar(gapminder, x="country", y="lifeExp")
    st.plotly_chart(fig, use_container_width=True)

def render_histogram_stacked():
    plt.clf()
    # Plot histogram based randomly-generating the results of rolling 2 dices
    dice_1 = np.random.randint(1,7,1000)
    dice_2 = np.random.randint(1,7,1000)
    dice_total = dice_1 + dice_2
    # bins = number of bars
    # defining x, color and label

    fig = px.histogram(dice_total, nbins=11, labels={'value':'Dice Roll'},
                title='5000 Dice Roll Histogram', marginal='violin',
                color_discrete_sequence=['green'])

    fig.update_layout(
        xaxis_title_text='Dice-Roll',
        yaxis_title_text='Dice-Total',
        bargap=0.2, showlegend=False
    )

    # Here we are stacking two histograms ontop of each other
    df_tips = px.data.tips()
    fig = px.histogram(df_tips, x="total_bill", color="sex")
    st.plotly_chart(fig, use_container_width=True)

def render_histogram_simple():
    plt.clf()
    # Simple Histogram
    df = px.data.gapminder().query("year == 2007")
    # Here we use a column with categorical data
    fig = px.histogram(df, x="gdpPercap")
    st.plotly_chart(fig, use_container_width=True)

def render_bar_chart_portugal_query():
    plt.clf()
    # Simple Bar chart with Portugal Query
    data_portugal = px.data.gapminder().query("country == 'Portugal'")
    fig = px.bar(data_portugal, x='year', y='pop')
    st.plotly_chart(fig, use_container_width=True)

def render_bar_chart_canada_colored():
    plt.clf()
    canada_data = px.data.gapminder().query(" country == 'Canada' ")
    fig = px.bar(canada_data, x= 'year', y='pop', 
       color = 'lifeExp', 
       height = 400, 
       labels= {'pop':'popuation'} )
    st.plotly_chart(fig, use_container_width=True)

def render_various_line_colors_styles():
    plt.clf()
    x=np.linspace(0,5,60)
    # Adjusting the Plot: Line Colors and Styles

    # The first adjustment you might wish to make to a plot is to control the line colors and styles. The plt.plot() function takes additional arguments that can be used to spec‐
    # ify these. To adjust the color, you can use the color keyword, which accepts a string argument representing virtually any imaginable color. The color can be specified in a variety of ways


    plt.plot(x, np.sin(x - 0), color='blue')      # specify color by name
    plt.plot(x, np.sin(x - 1), color='g')        # short color code (rgbcmyk)
    plt.plot(x, np.sin(x - 2), color='0.75')     # Grayscale between 0 and 1
    plt.plot(x, np.sin(x - 3), color='#FFDD44')      # Hex code (RRGGBB from 00 to FF)
    plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3))    # RGB tuple, values 0 and 1
    plt.plot(x, np.sin(x - 5), color='chartreuse');     # all HTML color names supported

    # If no color is specified, Matplotlib will automatically cycle through a set of default colors for multiple lines.

    # Similarly, you can adjust the line style using the linestyle keyword

    plt.plot(x, x + 0, linestyle='solid')
    plt.plot(x, x + 1, linestyle='dashed')
    plt.plot(x, x + 2, linestyle='dashdot')
    plt.plot(x, x + 3, linestyle='dotted')
        
    # For short, you can use the following codes:

    plt.plot(x, x + 4, linestyle='-') # solid
    plt.plot(x, x + 5, linestyle='--') # dashed
    plt.plot(x, x + 6, linestyle='-.') # dashdot
    plt.plot(x, x + 7, linestyle=':') # dotted
    
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_scatterplot_facets():
    plt.clf()
    # Several scatterplots are sometimes called, 'Facets'
    df_tips = px.data.tips()  #smoker tip df
    px.scatter(df_tips, x="total_bill", y="tip", color="smoker", facet_col="sex")

    # Setting plots into rows and columns
    px.histogram(df_tips, x="total_bill", y="tip", color="sex", facet_row="time", facet_col="day",
        category_orders={"day": ["Thur", "Fri", "Sat", "Sun", "Mon"], "time": ["Breakfast", "Lunch", "Dinner"]})


    att_df = sns.load_dataset("attention")
    fig = px.line(att_df, x='solutions', y='score', facet_col='subject',
                facet_col_wrap=5, title='Scores Based on Attention')
    st.plotly_chart(fig, use_container_width=True)
    
def render_2lines_1chart():
    plt.clf()
    x = np.linspace(1,10,200)
    plt.plot(x,np.sin(x))
    plt.plot(x,np.cos(x))
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_dashed_line_chart():
    plt.clf()
    x = np.linspace(0, 10, 100)
    fig=plt.figure()
    plt.plot(x,np.sin(x),'_')
    plt.plot(x,np.cos(x), '_')

    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_2lines_2chart():
    plt.clf()
    # one more way to draw graph

    plt.figure()
    x = np.linspace(0, 10, 100)
    # create the first of two panels and set current axis
    plt.subplot(2,1,1) # (rows, columns, panel number)
    plt.plot(x, np.sin(x))

    # create the second panel and set current axis

    plt.subplot(2,1,2)
    plt.plot(x, np.cos(x))
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_2lines_2colors_1chart():
    plt.clf()
    fig=plt.figure()
    ax=plt.axis()

    x=np.linspace(1,10,1000)

    x = np.linspace(0, 10, 2000)
    plt.plot(x, np.sin(x))

    plt.plot(x, np.sin(x))
    plt.plot(x, np.cos(x))
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_line_chart_life_expectancy():
    plt.clf()
    df = px.data.gapminder().query("country=='Lebanon'")
    fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Lebanon')
    st.plotly_chart(fig, use_container_width=True)

def render_map_plot_airbnb():
    plt.clf()
    df_AirBerlin = pd.read_csv("listings.csv")
    fig = px.density_mapbox(df_AirBerlin, lat = "latitude", lon ="longitude", z = "price", radius = 10, 
                       center = dict(lat = 9, lon =8), zoom = 1, hover_name = 'price', 
                       mapbox_style = 'open-street-map', title = 'Airbnb in Berlin Locations and Prices')
    st.plotly_chart(fig, use_container_width=True)

def render_world_population_top1000():
    plt.clf()
    us_cities = pd.read_csv("https://gist.githubusercontent.com/curran/13d30e855d48cdd6f22acdf0afe27286/raw/0635f14817ec634833bb904a47594cc2f5f9dbf8/worldcities_clean.csv")
    fig = px.scatter_mapbox(us_cities, lat="lat", lon="lng", hover_name="city", hover_data=["country", "city", "population"],
                            color_discrete_sequence=["orange"], zoom=3, height=300)
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

def render_density_mapbox():
    plt.clf()
    earthquake_df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv")
    fig = px.density_mapbox(earthquake_df, lat = "Latitude", lon ="Longitude", z = "Magnitude", radius = 10, 
                        center = dict(lat = 9, lon =8), zoom = 1, hover_name = 'Date', 
                        mapbox_style = 'stamen-watercolor', title = 'Earthquakes Years: 1965 to 2016')
    st.plotly_chart(fig, use_container_width=True)

def render_nuclear_waste_map():
    plt.clf()
    mapbox_access_token = "pk.eyJ1IjoibWFobW91ZHlhZ2htb3VyIiwiYSI6ImNrejl3aHN6dDA2M2EydnM2YWYzanoyZDIifQ.zUiYm5A4Ht3_zmnGkuD9gg"

    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/Nuclear%20Waste%20Sites%20on%20American%20Campuses.csv')
    site_lat = df.lat
    site_lon = df.lon
    locations_name = df.text

    data = [
        go.Scattermapbox(
            lat=site_lat,
            lon=site_lon,
            mode='markers',
            marker=dict(
                size=17,
                color='rgb(255, 0, 0)',
                opacity=0.7
            ),
            text=locations_name,
            hoverinfo='text'
        ),
        go.Scattermapbox(
            lat=site_lat,
            lon=site_lon,
            mode='markers',
            marker=dict(
                size=8,
                color='rgb(242, 177, 172)',
                opacity=0.7
            ),
            hoverinfo='none'
        )]


    layout = go.Layout(
        title='Nuclear Waste Sites on Campus',
        autosize=True,
        hovermode='closest',
        showlegend=False,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=38,
                lon=-94
            ),
            pitch=0,
            zoom=3,
            style='light'
        ),
    )

    fig = dict(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

def render_cheropleth_map():
    plt.clf()
    gapminder = px.data.gapminder()
    fig = px.choropleth(gapminder, locations = 'iso_alpha', color = 'lifeExp', hover_name = 'country', 
             animation_frame = 'year', color_continuous_scale =  px.colors.sequential.Plasma)
    st.plotly_chart(fig, use_container_width=True)

def render_cheropleth_map_natural_earth():
    plt.clf()
    gapminder = px.data.gapminder()
    fig = px.choropleth(gapminder, locations = 'iso_alpha', color = 'lifeExp', hover_name = 'country', 
             animation_frame = 'year', color_continuous_scale =  px.colors.sequential.Plasma, 
             projection = 'natural earth')
    st.plotly_chart(fig, use_container_width=True)

def render_scatter_geo_plot_orthographic_projection_iso():
    plt.clf()
    country_data = px.data.gapminder()
    map_fig = px.scatter_geo(country_data, 
                        locations = 'iso_alpha', 
                        projection = 'orthographic',
                        color = 'continent',
                        opacity = .8,
                        hover_name = 'country',
                        hover_data = ['gdpPercap', 'pop']
                        )
    st.plotly_chart(map_fig, use_container_width=True)

def render_scatter_geo_plot_orthographic_iso_lines():
    plt.clf()
    gapminder = px.data.gapminder()
    fig = px.line_geo(gapminder, locations="iso_alpha",
                  color="continent",
                  projection="orthographic")
    st.plotly_chart(fig, use_container_width=True)

def render_3d_plot_mt_bruno():
    plt.clf()
    z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

    fig = go.Figure(data=[go.Surface(z=z_data.values)])

    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                    width=600, height=600,
                    margin=dict(l=75, r=60, b=75, t=100))
    st.plotly_chart(fig)
    
def render_3d_plot():
    plt.clf()
    ax=plt.axes(projection='3d')
    # Below are data for a 3D line 

    zline=np.linspace(0,15,1000)
    yline=np.cos(zline)
    xline=np.sin(zline)

    ax.plot3D(xline,yline,zline,'red')

    ## Below are data for 3D scatter plot

    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_3d_contour_plot_black_white():
    plt.clf()
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X,Y=np.meshgrid(x,y)
    Z=f(X,Y)

    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #If default viewing angle isn't idea, then utilize view_init method to set the elevation and azimuthal angles. 

    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

    ax.view_init(60,35)
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)
    
def render_3d_contour_plot_yellow_wired():
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X,Y=np.meshgrid(x,y)
    Z=f(X,Y)
    plt.clf()
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, Z, color='Gold')
    ax.set_title('wireframe and Surface Plot')
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_3d_contour_plot_color_map():
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X,Y=np.meshgrid(x,y)
    Z=f(X,Y)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)   
    
def render_3d_contour_plot_partial_polar_color_map():
    plt.clf()
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))
    r=np.linspace(0,6,20)
    theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
    r,theta=np.meshgrid(r,theta)
    X=r*np.sin(theta)
    Y=r*np.cos(theta)
    Z=f(X,Y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')    
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_3d_contor_plot_sampling_color_map():
    plt.clf()
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))
    theta = 2 * np.pi * np.random.random(1000)
    r = 6 * np.random.random(1000)
    x = np.ravel(r * np.sin(theta))
    y = np.ravel(r * np.cos(theta))
    z = f(x, y)

    # We could create a scatter plot of the points to get an idea of the surface we’re sampling from

    ax=plt.axes(projection='3d')
    ax.scatter(x,y,z,c=z,cmap='viridis',linewidth=0.5)
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_3d_contor_plot_color_map_grid():
    plt.clf()
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))
    theta = 2 * np.pi * np.random.random(1000)
    r = 6 * np.random.random(1000)
    x = np.ravel(r * np.sin(theta))
    y = np.ravel(r * np.cos(theta))
    z = f(x, y)

    ax=plt.axes(projection='3d')
    ax.plot_trisurf(x,y,z,cmap='viridis',edgecolor='none')
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_3d_parameterization():
    plt.clf()
    def f(x,y):
        return np.sin(np.sqrt(x**2+y**2))
    theta = 2 * np.pi * np.random.random(1000)
    r = 6 * np.random.random(1000)
    x = np.ravel(r * np.sin(theta))
    y = np.ravel(r * np.cos(theta))
    z = f(x, y)
    from matplotlib.tri import Triangulation
    tri = Triangulation(np.ravel(r), np.ravel(theta))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles,cmap='viridis', linewidths=0.2)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_iris_3d_plot():
    plt.clf()
    df_iris = px.data.iris()
    fig = px.scatter_3d(df_iris, x='sepal_width', y='petal_width', z='sepal_length',
                color='species')
    st.plotly_chart(fig, use_container_width=True) 

def render_3d_plot_tips_df():
    plt.clf()
    df_tips = pd.read_csv("tips.csv")
    fig = px.scatter_3d(df_tips, x = 'total_bill', 
                    y = 'day', 
                    z = 'time',
                    color = 'sex', 
                    size='tip',
                    size_max = 20,
                    opacity = 0.7)
    st.plotly_chart(fig, use_container_width=True)

def render_3d_plot_with_lines():
    plt.clf()
    gapminder = px.data.gapminder()
    fig = px.line_3d(gapminder, x="gdpPercap", y="pop", z="year")
    st.plotly_chart(fig, use_container_width=True)

def render_iris_scatter_plot():
    plt.clf()
    df_iris = px.data.iris()
    px.scatter(df_iris, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_iris.sepal_width, y=df_iris.sepal_length,
        mode='markers',
        marker_color=df_iris.sepal_width,
        text=df_iris.species,
        marker=dict(showscale=True)
    ))
    fig.update_traces(marker_line_width=2, marker_size=10)


    fig = go.Figure(data=go.Scattergl(
        x = np.random.randn(100000),
        y = np.random.randn(100000),
        mode='markers',
        marker=dict(
            color=np.random.randn(100000),
            colorscale='Viridis',
            line_width=1
        )
    ))
    st.plotly_chart(fig, use_container_width=True)

def render_strip_chart():
    plt.clf()
    df = px.data.gapminder()
    df = pd.DataFrame(df)
    df = df.groupby(['year','continent'])['pop'].mean().reset_index()
    fig = px.strip(df, x="pop", y="year", color="continent", facet_col="continent")
    st.plotly_chart(fig, use_container_width=True)

def render_scatterplot_colored():
    plt.clf()
    gapminder = px.data.gapminder()
    gapminder2007 = gapminder.query('year==2007')
    fig = px.scatter(gapminder2007, x='gdpPercap', y= 'lifeExp', color = "continent")
    st.plotly_chart(fig, use_container_width=True)

def render_bubble_chart_life_expectancy():
    plt.clf()
    gapminder = px.data.gapminder()
    fig=px.scatter(gapminder, x= 'gdpPercap', y='lifeExp', color = 'continent', size = 'pop', 
          size_max = 60, hover_name = 'country', facet_col= 'continent', log_x = True, animation_frame = "year")
    st.plotly_chart(fig, use_container_width=True)

def render_animated_bubble_scatter_plot():
    plt.clf()
    gapminder = px.data.gapminder()
    fig=px.scatter(gapminder, x= 'gdpPercap', y='lifeExp', color = 'continent', size = 'pop', 
          size_max = 60, hover_name = 'country', log_x = True, 
           animation_frame = "year", 
           range_x = [25, 10000],
          range_y = [25, 90])
    st.plotly_chart(fig, use_container_width=True)

def render_scatter_plot_gapminder():
    plt.clf()
    df = px.data.gapminder()
    fig = px.scatter(df,x="pop", y="gdpPercap",
                    color="continent",
                    size='pop', hover_data=['country'],
                    title = 'Year-Population corellation')
    st.plotly_chart(fig, use_container_width=True)

def render_box_plot_whiskers():
    plt.clf()
    df = px.data.gapminder().query("year == 2007")
    # Here we use a column with categorical data
    fig = px.box(df, x="continent", y="gdpPercap")
    st.plotly_chart(fig, use_container_width=True)

def render_burst_pi_chart():
    plt.clf()
    #Pie Chart
    df_samer = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
    px.pie(df_samer, values='pop', names='country', 
        title='Population of Asian continent', 
        color_discrete_sequence=px.colors.sequential.RdBu)

    # Customize pie chart
    colors = ['blue', 'green', 'black', 'purple', 'red', 'brown']
    fig = go.Figure(data=[go.Pie(labels=['Water','Grass','Normal','Psychic', 'Fire', 'Ground'], 
                        values=[110,90,80,80,70,60])])
    # Define hover info, text size, pull amount for each pie slice, and stroke
    fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                    textinfo='label+percent', pull=[0.1, 0, 0.2, 0, 0, 0],
                    marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))
    st.plotly_chart(fig, use_container_width=True)

def render_sunburst_chart():
    plt.clf()
    df = px.data.gapminder().query("year == 2007")
    fig = px.sunburst(df, path=['continent','country','year'], values='pop', color='country',
                    color_discrete_map={'(?)':'black', 'India':'gold', 'China':'darkblue'})
    st.plotly_chart(fig, use_container_width=True)

def render_pie_chart():
    plt.clf()
    df = px.data.gapminder().query("year == 2007").query("continent == 'Africa'")
    df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' # Represent only large countries
    fig = px.pie(df, values='pop', names='country', title='Population of Africa continent')

    st.plotly_chart(fig, use_container_width=True)

def render_mulitple_violin_plot():
    plt.clf()
    df_tips = px.data.tips()
    px.violin(df_tips, y="total_bill", box=True, points='all')

    # plotting several plots
    px.violin(df_tips, y="tip", x="smoker", color="sex", box=True, points="all",
            hover_data=df_tips.columns)

    # Changing violin plots left or right based on if the client is a smoker
    fig = go.Figure()
    fig.add_trace(go.Violin(x=df_tips['day'][ df_tips['smoker'] == 'Yes' ],
                            y=df_tips['total_bill'][ df_tips['smoker'] == 'Yes' ],
                            legendgroup='Yes', scalegroup='Yes', name='Yes',
                            side='negative',
                            line_color='blue'))
    fig.add_trace(go.Violin(x=df_tips['day'][ df_tips['smoker'] == 'No' ],
                            y=df_tips['total_bill'][ df_tips['smoker'] == 'No' ],
                            legendgroup='Yes', scalegroup='Yes', name='No',
                            side='positive',
                            line_color='red'))
    st.plotly_chart(fig, use_container_width=True)

def render_simple_violin_plot():
    plt.clf()
    df = px.data.gapminder()
    fig = px.violin(df, y="lifeExp")
    st.plotly_chart(fig, use_container_width=True)

def render_tree_map_chart():
    plt.clf()
    df = px.data.gapminder().query("year == 2007")
    fig = px.treemap(df, path=['continent','country','year'], values='pop', color='country',
                 color_discrete_map={'(?)':'black', 'Brazil':'gold', 'Japan':'darkblue'})

    st.plotly_chart(fig, use_container_width=True)

def render_heat_map_golden_ratio_unequal_boxes():
    plt.clf()
    phi = (1 + np.sqrt(5) )/2. # golden ratio or fibonacci sequence
    xe = [0, 1, 1+(1/(phi**4)), 1+(1/(phi**3)), phi]
    ye = [0, 1/(phi**3), 1/phi**3+1/phi**4, 1/(phi**2), 1]

    z = [ [13,3,3,5],
        [13,2,1,5],
        [13,10,11,12],
        [13,8,8,8]
        ]

    fig = go.Figure(data=go.Heatmap(
            x = np.sort(xe),
            y = np.sort(ye),
            z = z,
            type = 'heatmap',
            colorscale = 'Plasma'))

    # Add spiral line plot

    def spiral(th):
        a = 1.120529
        b = 0.306349
        r = a*np.exp(-b*th)
        return (r*np.cos(th), r*np.sin(th))

    theta = np.linspace(-np.pi/13,4*np.pi,1000); # angle
    (x,y) = spiral(theta)

    fig.add_trace(go.Scatter(x= -x+x[0], y= y-y[0],
        line =dict(color='white',width=3)))

    axis_template = dict(range = [0,1.6], autorange = False,
                showgrid = False, zeroline = False,
                linecolor = 'black', showticklabels = False,
                ticks = '' )

    fig.update_layout(margin = dict(t=200,r=200,b=200,l=200),
        xaxis = axis_template,
        yaxis = axis_template,
        showlegend = False,
        width = 700, height = 700,
        autosize = False )
    st.plotly_chart(fig, use_container_width=True)

def render_filled_area_plot():
    plt.clf()
    gapminder = px.data.gapminder() # Viewing data for using below
    fig = px.area(gapminder, x="year", y="gdpPercap", color="continent",line_group="country")
    st.plotly_chart(fig, use_container_width=True)

def render_ternary_plot():
    plt.clf()
    df_experiment = px.data.experiment() #Viewing dataset for usage below - built-in
    fig=px.scatter_ternary(df_experiment, a="experiment_1", b="experiment_2", 
                   c='experiment_3', hover_name="gender", color="group")
    st.plotly_chart(fig, use_container_width=True)

def render_gantt_chart():
    plt.clf()
    df = pd.DataFrame([
    dict(Task="Job A", Start='2020-01-01', Finish='2020-02-28', Resource="Cleopatra"),
    dict(Task="Job B", Start='2020-03-05', Finish='2020-04-15', Resource="Mark"),
    dict(Task="Job C", Start='2020-02-20', Finish='2020-05-30', Resource="Caesar")
    ])

    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Resource")

    st.plotly_chart(fig, use_container_width=True)

def render_polar_chart():
    plt.clf()
    df_winddata = px.data.wind()
    px.scatter_polar(df_winddata, r="frequency", theta="direction", color="strength",
                    size="frequency", symbol="strength")
    fig=px.line_polar(df_winddata, r="frequency", theta="direction", color="strength",
                line_close=True, template="plotly_dark", width=800, height=400)
    st.plotly_chart(fig, use_container_width=True)

    buf = BytesIO()
    plt.gcf().savefig(buf, format="png")
    st.image(buf)

def render_latex_equations():
    def render_img(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf) 
    import matplotlib
    plt.clf()
    X = np.linspace(3, 8, 800)
    Y = X
    Y2 = X**3
    Y_log = np.log(X)
    Y_cos = np.cos(np.pi*X)

    matplotlib.rcParams['font.size'] = 20

    plt.plot(X, Y2, X, Y, X, Y_log)
    plt.xlabel('$x \in [1, \infty)$', fontsize=14)
    plt.ylabel('$y$',fontsize=14)
    plt.legend(['$y=x^2$', '$y=x$', '$y=\ln\;x$'], fontsize=12)
    plt.title('Comparison of $\mathcal{O}(x^2)$, $\mathcal{O}(x)$, and $\mathcal{O}(\ln\;x)$', fontsize=16)
    plt.text(3.4, 8, 'Rapid growth\nwhen $y \sim \mathcal{O}(x^2)$', fontsize=14, color='gray')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False);

    render_img(plt.gcf())

    plt.clf()
    plt.plot(X, Y2)
    plt.title(r'$y=x^2$');
    render_img(plt.gcf())

    plt.clf()
    # Symbols and Sub/Superscripts
    plt.plot(X, Y)
    plt.title(r'$y=x_i$');
    render_img(plt.gcf())

    plt.clf()
    # Symbols and Sub/Superscripts
    plt.plot(X, Y)
    plt.title(r'$y=x_{base}$');  
    render_img(plt.gcf())

    plt.clf()
    plt.plot(X, Y_cos)
    plt.title(r'$y=\cos(\pi x)$')
    plt.text(2.8, 1.25, r'$y=\cos(\pi x)$')
    plt.xlabel(r'$y=\cos(\pi x)$')
    plt.ylabel(r'$y=\cos(\pi x)$')
    plt.legend([r'$y=\cos(\pi x)$'], loc=4); 
    render_img(plt.gcf())

    plt.clf()
    x = np.linspace(0.5, 100, 1000)
    y_frac = 2 + 1/(x**2);

    plt.plot(x, y_frac)
    plt.ylim(0, None)
    plt.text(40, 4, r'$\lim_{x\rightarrow\infty}\;2 + \frac{1}{x^2} = 2$', fontsize=20);
    render_img(plt.gcf())

    plt.clf()
    plt.plot(X, Y_log)
    plt.title(r'$\ddot{U}$ber is German for Super');
    render_img(plt.gcf())

def render_3d_parametric_plot():
    plt.clf()
    s = np.linspace(0, 2 * np.pi, 240)
    t = np.linspace(0, np.pi, 240)
    tGrid, sGrid = np.meshgrid(s, t)

    r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
    x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
    y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
    z = r * np.cos(tGrid)                  # z = r*cos(t)

    surface = go.Surface(x=x, y=y, z=z)
    data = [surface]

    layout = go.Layout(
        title='Parametric Plot',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)
    
def render_animated_figure_dash():
    plt.clf()
    df = px.data.gapminder()
    fig = px.scatter(
        df, x="gdpPercap", y="lifeExp", animation_frame="year", 
        animation_group="country", size="pop", color="continent", 
        hover_name="country", log_x=True, size_max=55, 
        range_x=[100,100000], range_y=[25,90])
    st.plotly_chart(fig, use_container_width=True)

def render_animated_bar_chart_plotly():
    plt.clf()
    df = px.data.gapminder()
    fig = px.bar(df, x="continent", y="pop", color="continent", 
    animation_frame="year", animation_group="country", range_y=[0,4000000000])
    st.plotly_chart(fig, use_container_width=True)

def render_simple_button():
    plt.clf()
    fig = go.Figure(
    data=[go.Scatter(x=[0, 1], y=[0, 1])],
    layout=go.Layout(
        xaxis=dict(range=[0, 5], autorange=False),
        yaxis=dict(range=[0, 5], autorange=False),
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),
            go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])]),
            go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])],
                     layout=go.Layout(title_text="End Title"))]
    )
    st.plotly_chart(fig, use_container_width=True)

def render_moving_point_curve():
    plt.clf()
    # Generate curve data
    t = np.linspace(-1, 1, 100)
    x = t + t ** 2
    y = t - t ** 2
    xm = np.min(x) - 1.5
    xM = np.max(x) + 1.5
    ym = np.min(y) - 1.5
    yM = np.max(y) + 1.5
    N = 50
    s = np.linspace(-1, 1, N)
    xx = s + s ** 2
    yy = s - s ** 2


    # Create figure
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y,
                        mode="lines",
                        line=dict(width=2, color="blue")),
            go.Scatter(x=x, y=y,
                        mode="lines",
                        line=dict(width=2, color="blue"))],
        layout=go.Layout(
            xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
            title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
            updatemenus=[dict(type="buttons",
                            buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None])])]),
        frames=[go.Frame(
            data=[go.Scatter(
                x=[xx[k]],
                y=[yy[k]],
                mode="markers",
                marker=dict(color="red", size=10))])

            for k in range(N)]
    )    
    st.plotly_chart(fig, use_container_width=True)

def render_moving_frenet_frame():
    plt.clf()
    # Generate curve data
    t = np.linspace(-1, 1, 100)
    x = t + t ** 2
    y = t - t ** 2
    xm = np.min(x) - 1.5
    xM = np.max(x) + 1.5
    ym = np.min(y) - 1.5
    yM = np.max(y) + 1.5
    N = 50
    s = np.linspace(-1, 1, N)
    xx = s + s ** 2
    yy = s - s ** 2
    vx = 1 + 2 * s
    vy = 1 - 2 * s  # v=(vx, vy) is the velocity
    speed = np.sqrt(vx ** 2 + vy ** 2)
    ux = vx / speed  # (ux, uy) unit tangent vector, (-uy, ux) unit normal vector
    uy = vy / speed

    xend = xx + ux  # end coordinates for the unit tangent vector at (xx, yy)
    yend = yy + uy

    xnoe = xx - uy  # end coordinates for the unit normal vector at (xx,yy)
    ynoe = yy + ux


    # Create figure
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y,
                        name="frame",
                        mode="lines",
                        line=dict(width=2, color="blue")),
            go.Scatter(x=x, y=y,
                        name="curve",
                        mode="lines",
                        line=dict(width=2, color="blue"))
            ],
        layout=go.Layout(width=600, height=600,
                        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
                        title="Moving Frenet Frame Along a Planar Curve",
                        hovermode="closest",
                        updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])]),

        frames=[go.Frame(
            data=[go.Scatter(
                x=[xx[k], xend[k], None, xx[k], xnoe[k]],
                y=[yy[k], yend[k], None, yy[k], ynoe[k]],
                mode="lines",
                line=dict(color="red", width=2))
            ]) for k in range(N)]
    )
    st.plotly_chart(fig, use_container_width=True)

def render_slider_with_buttons():
    plt.clf()
    url = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
    dataset = pd.read_csv(url)

    years = ["1952", "1962", "1967", "1972", "1977", "1982", "1987", "1992", "1997", "2002",
            "2007"]

    # make list of continents
    continents = []
    for continent in dataset["continent"]:
        if continent not in continents:
            continents.append(continent)
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [30, 85], "title": "Life Expectancy"}
    fig_dict["layout"]["yaxis"] = {"title": "GDP per Capita", "type": "log"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Year:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    year = 1952
    for continent in continents:
        dataset_by_year = dataset[dataset["year"] == year]
        dataset_by_year_and_cont = dataset_by_year[
            dataset_by_year["continent"] == continent]

        data_dict = {
            "x": list(dataset_by_year_and_cont["lifeExp"]),
            "y": list(dataset_by_year_and_cont["gdpPercap"]),
            "mode": "markers",
            "text": list(dataset_by_year_and_cont["country"]),
            "marker": {
                "sizemode": "area",
                "sizeref": 200000,
                "size": list(dataset_by_year_and_cont["pop"])
            },
            "name": continent
        }
        fig_dict["data"].append(data_dict)

    # make frames
    for year in years:
        frame = {"data": [], "name": str(year)}
        for continent in continents:
            dataset_by_year = dataset[dataset["year"] == int(year)]
            dataset_by_year_and_cont = dataset_by_year[
                dataset_by_year["continent"] == continent]

            data_dict = {
                "x": list(dataset_by_year_and_cont["lifeExp"]),
                "y": list(dataset_by_year_and_cont["gdpPercap"]),
                "mode": "markers",
                "text": list(dataset_by_year_and_cont["country"]),
                "marker": {
                    "sizemode": "area",
                    "sizeref": 200000,
                    "size": list(dataset_by_year_and_cont["pop"])
                },
                "name": continent
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [year],
            {"frame": {"duration": 300, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": year,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)

