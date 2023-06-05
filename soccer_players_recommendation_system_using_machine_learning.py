# -*- coding: utf-8 -*-
"""Soccer_Players_Recommendation_System_Using_Machine_Learning.ipynb
"""

!pip install JayDeBeApi

import numpy as np
import pandas as pd

import jaydebeapi

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

"""## Loading Data"""

df = pd.read_csv('data.csv')

df.head()

df.shape

df.isnull().sum()

df.columns

df = df.dropna()

df.isnull().sum().sum()

df.dtypes

def create_players_table(curs):
    curs.execute("""
    CREATE TABLE IF NOT EXISTS players (
        name VARCHAR(255),
        club VARCHAR(255),
        age INTEGER,
        position VARCHAR(255),
        position_cat INTEGER,
        market_value DOUBLE,
        page_views INTEGER,
        fpl_value DOUBLE,
        fpl_sel VARCHAR(255),
        fpl_points INTEGER,
        region DOUBLE,
        nationality VARCHAR(255),
        new_foreign INTEGER,
        age_cat INTEGER,
        club_id INTEGER,
        big_club INTEGER,
        new_signing INTEGER
    )
    """)

def load_data_to_griddb(conn, data_frame = df):
    data = df
    
    for index, row in data.iterrows():
        values = tuple(row.values)
        curs.execute("INSERT INTO players (name, club, age, position, position_cat, market_value, "
                     "page_views, fpl_value, fpl_sel, fpl_points, region, nationality, new_foreign, "
                     "age_cat, club_id, big_club, new_signing) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)

def query_sensor(curs, table):
    curs.execute("select name, club, age, position, position_cat, market_value, "
                 "page_views, fpl_value, fpl_sel, fpl_points, region, nationality, new_foreign, "
                 "age_cat, club_id, big_club, new_signing from " + table)
    return curs.fetchall()[0][0]

url = "jdbc:gs://" + "239.0.0.1" + ":" + "41999" + "/" + "defaultCluster"
conn = jaydebeapi.connect("com.toshiba.mwcloud.gs.sql.Driver",
    url,  ["admin", "admin"], "./gridstore-jdbc.jar")

curs = conn.cursor()
create_players_table(curs) 
load_data_to_griddb(conn)
curs.execute("select table_name from \"#tables\"")
tables = []
data = []

for table in curs.fetchall():
    try:
        if table[0] == "players":
            tables.append(table[0])
            data.append(query_sensor(curs, table[0]))
    except:
        pass

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