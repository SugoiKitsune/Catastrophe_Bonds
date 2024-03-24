# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:17:03 2024

@author: Andrey
"""

import json
import requests
#import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
#import ttwick
import time
import locale
import Circles

from scipy.stats import percentileofscore
#from pandas.tools.plotting import scatter_matrix
from geopy.geocoders import ArcGIS
from circles import Circles
from datetime import datetime
from dateutil import parser
from io import BytesIO
#from IPython.utils.trailets import Unicode
from IPython.display import HTML
from mlp_toolkits.basemap import *
from math import radians, sqrt, sin, cos, atan2


locale.setlocale(locale.LC_ALL, '')
#sns.set_palette('deep', desat = .6)
sns.set(style = 'darkgrid')
sns.set_palette('muted', 7)
pd.options.display.float_format = '{z,.4f}'.format

city_state_country = 'San Salvador, El Salvador'
geolocator = ArcGIS(timeout=50)
location = geolocator.geocode(city_state_country)
latitude = location.latitude
longitude = location.longitude


geolocator = ArcGIS(timeout=10)
location = geolocator.geocode(city_state_country)
z = '<iframe src=http://maps.yahoo.com/#mvt=M&lat='+str(latitude)+'&lon='+str(longitude)

HTML(z)



def fetch_earthquake_data(start_date, end_date, min_magnitude=0, max_results=20000, latitude=None, longitude=None, max_radius=None):
    """
    Fetches earthquake data from the USGS API and returns it as a Pandas DataFrame.
    
    Parameters:
    - start_date (str): Start date for the earthquake data (format: "YYYY-MM-DD").
    - end_date (str): End date for the earthquake data (format: "YYYY-MM-DD").
    - min_magnitude (float): Minimum magnitude of earthquakes to retrieve (default: 0).
    - max_results (int): Maximum number of earthquake results to retrieve (default: 20000).
    - latitude (float): Latitude of the center for the search area (default: None).
    - longitude (float): Longitude of the center for the search area (default: None).
    - max_radius (float): Maximum radius from the center to search for earthquakes, in degrees (default: None).
    
    Returns:
    - Pandas DataFrame: DataFrame containing earthquake data.
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    # Parameters for the API request
    parameters = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_magnitude,
        "limit": max_results
    }
    
    # Adding location parameters if specified
    if latitude is not None and longitude is not None:
        parameters["latitude"] = latitude
        parameters["longitude"] = longitude
        if max_radius is not None:
            parameters["maxradiuskm"] = max_radius
    
    # Sending GET request to the USGS API
    response = requests.get(base_url, params=parameters)
    
    if response.status_code == 200:
        # Extracting earthquake data from the response
        data = response.json()
        features = data['features']
        
        # Extracting relevant information for each earthquake
        earthquake_data = []
        for feature in features:
            properties = feature['properties']
            mag = properties['mag']
            place = properties['place']
            time = properties['time']
            lon, lat, _ = feature['geometry']['coordinates']
            
            earthquake_data.append({
                'Magnitude': mag,
                'Place': place,
                'Time': pd.to_datetime(time, unit='ms'),
                'Longitude': lon,
                'Latitude': lat
            })
        
        # Creating a DataFrame from the extracted earthquake data
        df = pd.DataFrame(earthquake_data)
        return df
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
        return None


def map_universe():
    geolocator = ArcGIS(timeout= 10)
    location =  geolocator.geocode(city_state_country)
    z = '<iframe src = http://maps.yahoo.com/#'




def geocalc(lat1, lat2, lon1, lon2):
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
        
    dlon = lon1 - lon2
    
    EARTH_R = 6372
    y = sqrt((cos(lat2)*sin(dlon))**2+(cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(dlon))**2)
    x = sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(dlon)
    c = atan2(y,x)
    
    return EARTH_R*c

def calc_distance(row):
    return geocalc(row['Latitude'], row['Longitude'], latitude, longitude)

def filter_dataframe(earthquake_df):
    #earthquake_df.describe()    
    
    min_mag_plot = 5
    filtered_df = earthquake_df[earthquake_df['Magnitude'] >= min_mag_plot]
    filtered_df  = filtered_df.sort_values('Magnitude', ascending=False)
    top_10 = filtered_df.head(10)
    
    
    
# Example usage:
start_date = "1900-01-01"
end_date = "2024-02-25"
max_radius = 500
earthquake_df = fetch_earthquake_data(start_date, end_date, max_radius, latitude, longitude)




def bond_pricing_model(magnitude_s):
    """`xx`x`
    Lane-Sanchez model for spread computation on catastrophe bond
    """
    gamma = 0.2817
    alpha = 0.4059
    beta = 0.1934     
    
    scores =np.linspace(6.5,8.0,16)
    percentiles = [100 - percentileofscore(magnitude_s.to_list(), score) for scores]
    percentile_df = pd.DataFrame({'PFL':percentiles}, index=scores)
    percentile_df['PFL'] = percentile_df['PFL']/100 
    
    pfl = 0.08
    
    
    time_series_lower_limit = '1978-1-1 00:00:00'
    time_series_upper_limit = '2003-12-31 00:00:00'
    time_df = earthquake_df[(earthquake_df.index>time_series_lower_limit) & (earthquake_df.index<time_series_upper_limit)]
    time_df.describe()
    
    

def monte_carlo_simulation(latitude_s, longitude_s, magnitude_s, distance_s):
    sims = 1000000
    sress = 0.05
    mu = time_df['Magnitude'].mean()
    sigma = time_df['Magnitude'].std()
    
    sim_latitude = np.random.choice(latitude_s, sims, replace=True)
    sim_latitude_1 = sim_latitude.tolist()
    
    sim_longitude = np.random.choice(longitude_s, sims, replace=True)
    sim_longitude_1 = sim_longitude.tolist()
    
    sim_magnitude = (1+stress)*np.random.choice(magnitude_s, sims, replace=True)
    sim_magnitude_1 = sim_magnitude.tolist()
    
    sim_distance = np.random.choice(distance_s, sims, replace=True)
    sim_distance_1 = sim_distance.tolist()
    
    
    data1 = sim_magnitude
    data2 = earthquake_df['Magnitude']
    
    plt.hist(data1, 40, normed=True, color=c1, alpha=.20)
    plt.hist(data2, 40, normed=True, color=c2, alpha=.20)


    first_ring_hits =[]
    second_ring_hits =[]
    third_ring_hits =[]
    outside_hits =[]

    for sim in range(sims):
        if sim_distance_1[sim] < first_circle:
            first_ring_hits.append(sim_magnitude[sim])
        elif sim_distance_1[sim] < second_circle and sim_distance_1[sim] >= first_ring_hits[sim]:
            secon_ring_hits.append(sim_magnitude[sim])
        elif sim_distance_1[sim] < third_circle and sim_distance_1[sim] >= secon_ring_hits[sim]:
            third_ring_hits.append(sim_magnitude[sim])
        else:
            outside_hits.append(sim_magnitude[sim])












    Returns
    -------
    None.

    """
    
    gamma = 0.2817



# Displaying the DataFrame
print(earthquake_df.head()) 