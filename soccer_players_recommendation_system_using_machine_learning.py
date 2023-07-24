# -*- coding: utf-8 -*-
"""Soccer_Players_Recommendation_System_Using_Machine_Learning

## GridDB Installation
"""

!wget https://github.com/griddb/c_client/releases/download/v5.3.0/griddb-c-client_5.3.0_amd64.deb

!dpkg -i griddb-c-client_5.3.0_amd64.deb

!pip install swig

!pip install griddb-python

"""## Import Libraries"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import griddb_python as griddb

"""## Loading Data"""

data = pd.read_csv('data.csv')

data.head(3)

data.shape

data.isnull().sum()

data.columns

data = data.dropna()

data.isnull().sum().sum()

data.dtypes

data.tail()

"""## Setting Up a Container in Griddb to Store the Data"""

def griddb_CRUD():
    factory = griddb.StoreFactory.get_instance()

    # Provide the necessary arguments
    gridstore = factory.get_store(
        host = '127.0.0.1',
        port = 10001,
        cluster_name = 'defaultCluster',
        username = 'admin',
        password = 'admin'
    )

    # Define the container info
    conInfo = griddb.ContainerInfo(
        "football_players",
        [
            ["name", griddb.Type.STRING],
            ["club", griddb.Type.STRING],
            ["age", griddb.Type.INTEGER],
            ["position", griddb.Type.STRING],
            ["position_cat", griddb.Type.INTEGER],
            ["market_value", griddb.Type.DOUBLE],
            ["page_views", griddb.Type.INTEGER],
            ["fpl_value", griddb.Type.DOUBLE],
            ["fpl_sel", griddb.Type.STRING],
            ["fpl_points", griddb.Type.INTEGER],
            ["region", griddb.Type.INTEGER],
            ["nationality", griddb.Type.STRING],
            ["new_foreign", griddb.Type.INTEGER],
            ["age_cat", griddb.Type.INTEGER],
            ["club_id", griddb.Type.INTEGER],
            ["big_club", griddb.Type.INTEGER],
            ["new_signing", griddb.Type.INTEGER]
        ],
        griddb.ContainerType.COLLECTION, True
    )

    # Drop container if it exists
    gridstore.drop_container(conInfo.name)

    # Create a container
    container = gridstore.put_container(conInfo)

    # Load the data
    data = pd.read_csv('data.csv')

    # Put rows
    for i in range(len(data)):
        row = data.iloc[i].tolist()
        container.put(row)

    # Get rows
    columns = ', '.join(data.columns)
    query = container.query(f"SELECT {columns}")
    rs = query.fetch(False)

    data_list = []
    while rs.has_next():
        data = rs.next()
        data_list.append(data)

    # Convert the list to a DataFrame
    df = pd.DataFrame(data_list, columns=data.columns)

    return df

data = griddb_CRUD()

"""## Recommendation System"""

sample = data.select_dtypes(include='number')
corr = sample.corr()
mask = np.zeros_like(corr, dtype = np.bool_)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10))
sns.heatmap(corr, mask=mask)

scaled = StandardScaler()
X = scaled.fit_transform(sample)
recommendations = NearestNeighbors(n_neighbors = 5, algorithm='kd_tree')
recommendations.fit(X)
player_index = recommendations.kneighbors(X)[1]

def find_index(x):
    return data[data['name']==x].index.tolist()[0]

def recommendation_system(player):
    print("Here are four players who are similar to {}: ".format(player))
    index =  find_index(player)
    for i in player_index[index][1:]:
        print("Player Name: {}\nPlayer Market Value: €{}\nPlayer Age: {}\nPlayer Current Club: {}\n".format(data.iloc[i]['name'],
                                                                                        data.iloc[i]['market_value'],
                                                                                        data.iloc[i]['age'],
                                                                                        data.iloc[i]['club']))

recommendation_system('Petr Cech')