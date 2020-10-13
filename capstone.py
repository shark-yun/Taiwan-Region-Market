#!/usr/bin/env python
# coding: utf-8

# <h2>1.1 Introduction

# Welcome to my final project notebook. This notebook will delve deep into the neighborhoods of Taiwan and, using the Foursquare API, find venues in each neighborhood. After gathering all of the required data, analysis will follow to help cluster the different neighborhoods and gain insight into which area have what top 3 common venue.

# In[ ]:





# 
# 
# 1.2 Business Problem
# A client would like to open a new business, however they do not know what kind of business they would like to open, let alone where to open it. In this project, I will determine the optimal areas in Taiwan to open a business, as well as the perfect business for that area.
# 
# 1.3 Data Gathering Method
# In order to answer the business question that I set up from previous notebook
# 
# I will be gathering data from a few sources.
# 
# The first source will be Wikipedia. My project is about the regions of Taiwan. Next, using Google Maps' API, we will collect the approximate coordinates of each area. With the coordinates and area names collected, we will next be using the Foursquare API to collect venue information for each area, within a designated radius. The venue data collected from Foursquare will then be used to determine the top venues in each area.
# 
# Once the venue information is gathered, the next step will be to cluster areas in Taiwan based on venues categories. This information will allow us to cluster customers in each area before moving on to identify areas within the clusters which are prime candidates for a new venue, as well as identifying which specific venues would be the most lucrative.
# 
# 2.1 Data Collection
# 2.1.1 Import Library

# In[1]:


import pandas as pd
import numpy as np
import urllib.request, urllib.parse, urllib.error
import ssl
from bs4 import BeautifulSoup
import re


# In[2]:


get_ipython().system(' pip install geocoder')
get_ipython().system('conda install tensorflow')
get_ipython().system('conda install -c conda-forge geopy --yes')
import geocoder
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values


# In[3]:



get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
get_ipython().system('pip install bs4')



import json
import requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans # import k-means from clustering stage
import folium # map rendering library
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score


# In[4]:


ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


# In[5]:


url = "https://en.wikipedia.org/wiki/Regions_of_Taiwan"
fhand = urllib.request.urlopen(url, context=ctx).read()
soup = BeautifulSoup(fhand, 'lxml')


# In[7]:


df = pd.DataFrame()


# In[8]:


table_1 = soup.find_all('table')[1]
table_2 = soup.find_all('table')[2]
table_3 = soup.find_all('table')[3]


df_1 = pd.read_html(str(table_1))
df_2 = pd.read_html(str(table_2))
df_3 = pd.read_html(str(table_3))

df = pd.concat([df_1[0], df_2[0], df_3[0]], axis=0)


# In[9]:


df.shape


# In[10]:


df_tw = df.copy()


# In[11]:


df_tw.reset_index(drop=True, inplace=True)
df_tw = df_tw[['Present divisions']]
df_tw.rename(columns={'English':'Area'}, inplace=True)
df_tw.head()


# In[12]:


df_tw.dropna(axis=0, inplace=True)
df_tw.shape


# In[35]:


import requests
api_key='Put_My_Api'


# In[36]:



def get_coordinates(api_key, address, verbose=False):
    try:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(api_key, address)
        response = requests.get(url).json()
        if verbose:
            print('Google Maps API JSON result =>', response)
        results = response['results']
        geographical_data = results[0]['geometry']['location'] # get geographical coordinates
        lat = geographical_data['lat']
        lon = geographical_data['lng']
        return [lat, lon]
    except:
        return [None, None]


# In[37]:



latitudes = []
longitudes = []
i=0

# loop until you get all the coordinates
while i < len(df_tw):
    address = df_tw['Present divisions'].iloc[i] + ', Taiwan'
    g = get_coordinates(api_key, address)
    latitudes.append(g[0])
    longitudes.append(g[1])
    print(' . ', end='')
    i += 1
print('Done')


# In[38]:


df_tw['Latitude'] = latitudes
df_tw['Longitude'] = longitudes
df_tw.head()


# In[39]:


count = 0
for i, v in enumerate(df_tw['Latitude']):
    if v == None:
        count += 1
print('There are {} missing values'.format(count))


# In[40]:


df_tw.to_csv('df_tw.csv')


# In[41]:


address = 'Taipei, Taiwan'
g = get_coordinates(api_key, address)
latitude = g[0]
longitude = g[1]
print('The coordinates of {} are: {}, {}'.format(address, latitude, longitude))


# In[44]:



# create map of Taiwan using latitude and longitude values
map_taiwan = folium.Map(location=[latitude, longitude], zoom_start=9.5)

# add markers to map
for area, lat, lng in zip(df_tw['Present divisions'], df_tw['Latitude'], df_tw['Longitude']):
    label = '{}'.format(area)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_taiwan)  

map_taiwan


# In[48]:


CLIENT_ID = 'FUS1N4A2ALEKEGVDFL1Z21RBME1X3AA15NCTFPVACGTI4EN3' 
CLIENT_SECRET = 'NNHKRPZ0DUYJ54TFHBZ3KO3MA2APISVBR3TB4Q12TL4PMSA2' 
VERSION = '20180604' 
LIMIT = 100


# In[49]:



def getNearbyVenues(names, latitudes, longitudes, radius=500):
    LIMIT = 100
    radius = 5000
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(' . ', end='')
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Area', 
                  'Area Lat', 
                  'Area Long', 
                  'Venue', 
                  'Venue Lat', 
                  'Venue Long', 
                  'Category']
    print('done')
    return(nearby_venues)


# In[50]:


taiwan_venues = getNearbyVenues(names = df_tw['Present divisions'], latitudes = df_tw['Latitude'], longitudes = df_tw['Longitude'])


# In[51]:


taiwan_venues.head()


# In[52]:


taiwan_venues.groupby('Area').count()


# In[53]:


print('There are {} uniques categories.'.format(len(taiwan_venues['Category'].unique())))


# In[54]:


taiwan_onehot = pd.get_dummies(taiwan_venues[['Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
taiwan_onehot['Area'] = taiwan_venues['Area'] 

# move neighborhood column to the first column
fixed_columns = [taiwan_onehot.columns[-1]] + list(taiwan_onehot.columns[:-1])
taiwan_onehot = taiwan_onehot[fixed_columns]


# In[55]:


taiwan_onehot.shape


# In[56]:


taiwan_grouped = taiwan_onehot.groupby('Area').mean().reset_index()


# In[57]:


taiwan_grouped_clustering = taiwan_grouped.drop('Area', 1)


# In[76]:


sse = {}
list_1 = []
list_2 = []
sil = pd.DataFrame()

for k in range(3,10):
    kmeans = KMeans(n_clusters=k, max_iter=10000, random_state=0).fit(taiwan_grouped_clustering)
    taiwan_grouped_clustering["Clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    label = kmeans.labels_
    sil_coeff = silhouette_score(taiwan_grouped_clustering, label, metric='euclidean')
    list_1.append(k)
    list_2.append(sil_coeff)

sil['k'] = list_1
sil['sil_coeff'] = list_2
highest = sil.sort_values(['sil_coeff'], ascending=False).head(1)
print('The best k is {} with a silhouette score of {}'.format(highest['k'].values, highest['sil_coeff'].values))
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[77]:


kclusters = 4

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0, max_iter=10000).fit(taiwan_grouped_clustering)


# In[78]:


kmeans.labels_[0:10]


# In[79]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[83]:



num_top_venues = 3
indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Area']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Area'] = taiwan_grouped['Area']

for ind in np.arange(taiwan_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(taiwan_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# In[ ]:

#加入Reviews, 對比2個Block內同行




