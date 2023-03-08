# **Soccer Players Recommendation System Using Machine Learning, Python, and GridDB**

The soccer business is a multi-billion dollar industry combining high-performing athletes, passionate fans, and big sponsorship deals. Team owners and managers around the globe are always looking for an edge to find the best talent that can add to the winning soccer o their team. Using machine learning to help team managers find a balanced, talented team is a perfect combination of technology and data to add value to a business.
This article will cover recommendation system models to help managers find talent and upcoming players based on their performance and soccer league data. The objective is to use the recommendation system as a monitoring and prospecting tool to find innovative, high-performing soccer players. In this article, we propose a recommendation system model to recommend soccer players given their soccer matches data using Python and GridDB.

## **Setting up your environment**

To implement the recommendation system described in this article, we begin by configuring your machine's environment to execute the Python code properly. Below are some of the prerequisites that must be met in your environment:

- **Windows 11:** For our solution, [Windows 11](https://www.microsoft.com/software-download/windows11) is used as our operating system.
- **GridDB:**  [GridDB](https://docs.griddb.net/latest/gettingstarted/python/#installation) is our database that stores the data used in the recommendation system model.
- **Python 3.11.2:** The latest version of [Python 3.11.2](https://www.python.org/downloads/) is used in our solution.
- **Jupyter Notebook:** [Jupyter Notebook](https://jupyter.org/install) is an integrated development environment (IDE) to run our Python code.
- **Visual Studio 2022 version 17.5:** [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) will run our GridDB database commands using JPype1 on Jupyter Notebook.

Finally, we must install any missing libraries utilized in Jupyter Notebook. We are missing [JayDeBeApi](https://pypi.org/project/JayDeBeApi/) in our situation. Here's how we use the pip command to install the missing library in the Jupyter terminal:

```bash
pip install JayDeBeApi
```

We may now explore our dataset after successfully installing and configuring our environment.

## **Introduction to the dataset** 

The dataset used in this article contains **416** rows and **17** columns. The dataset comprises attributes defining soccer players in terms of their role and historical accomplishments. These attributes consider the current market value, field position, and points scored. 

The following is the list of the features that are found in our dataset:

- **Name:** Player name in a text value.
- **Club:** Club name in a text value.
- **Age:** Player age in numerical value.
- **Position:** Player position in the field in a text value.
- **Position Category:** A categorical variable representing the player's field position.
- **Market Value:** Player market price in numerical value.
- **Page Views:** The number of Wikipedia page views is calculated as a daily average.
- **Fantasy League Value:** Player Fantasy League Price in numerical value.
- **Fantasy League Selection:** Player Fantasy League Selection in numerical value.
- **Fantasy League Points:** Player Fantasy League Points in numerical value.
- **Region:** A categorical variable representing the region of the player.
- **Nationality:** A text value that represents the nationality of the player.
- **New Foreign:** A boolean value to indicate if the players signed up with a foreign club newly.
- **Age Category:** The age group of the player.
- **Club ID:** A numerical value that represents a club identifier.
- **Big Club:** A boolean value to indicate if the players signed up with a big club or not.
- **New Signing:** A boolean value to tell if the players signed up with a club newly.

The dataset was extracted from the [English Premier League Players Dataset](https://www.kaggle.com/datasets/mauryashubham/english-premier-league-players-dataset). The table below is the first **three** rows of this dataset:

![image-20230304035905725](C:\Users\vip phone\AppData\Roaming\Typora\typora-user-images\image-20230304035905725.png)

## **Importing the necessary libraries** 

In this article, we will be using multiple Python modules that we will import according to their usage to build our recommendation system:

- Python libraries used to read and preprocess the dataset:

  ```python
  import numpy as np 
  import pandas as pd
  ```

- Python libraries are used to explore the dataset using graphs and plots:

  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  ```
  
- Python libraries used to build the recommendation system model:

  ```python
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import NearestNeighbors
  from sklearn.decomposition import PCA
  ```


After successfully importing the required libraries, we begin with reading our **English Premier League Players Dataset**.

## **Loading the Dataset** 

GridDB plays a significant role in creating our recommendation system as it is the data storage mechanism we will use to store our dataset. To successfully store the data, we will first load the GridDB container. This can be done using the `jaydebeapi` library we installed earlier. Next, we will use the **GridDB JDBC** connector to insert the data using a loop. Once done, we must load the data back into a data frame to continue creating our recommendation system.

The code described in this section can be written as follows:

```python
import jaydebeapi

def query_sensor(curs, table):
    curs.execute("select count(*) from " + table)
    return curs.fetchall()[0][0]

conn = jaydebeapi.connect("com.toshiba.mwcloud.gs.sql.Driver", 
                          "jdbc:gs://239.0.0.1:41999/defaultCluster", 
                          ["admin", "admin"], 
                          "gridstore-jdbc-4.6.0.jar")

curs = conn.cursor()
curs.execute("select * from \"#tables\"")
tables = []
data = []

for table in curs.fetchall():
    try:
        if table[1].split("_")[1].startswith("M"):
            tables.append(table[1])
            data.append(query_sensor(curs, table[1]))
    except:
        pass
```

## **Exploratory Data Analysis** 

Before we build our recommendation system, we must begin with an exploratory data analysis that will allow us to find any inconsistencies in our data and overall visualization of the dataset.
First, we begin by checking for any null values in our attributes. This is achieved with the following lines of code:

```python
data.isnull().sum()
```

This cell outputs the following results, indicating that we have **one** missing value for the **region** attribute: 

```python
name            0
club            0
age             0
position        0
position_cat    0
market_value    0
page_views      0
fpl_value       0
fpl_sel         0
fpl_points      0
region          1
nationality     0
new_foreign     0
age_cat         0
club_id         0
big_club        0
new_signing     0
dtype: int64
```

To clean up our missing value, we can use the built-in method `dropna()`. This method returns the newly cleaned dataset to replace the old uncleaned one. 

This is achieved with the following code:

```python
data = data.dropna()
```

Now we have replaced all missing values; we can move to visualize our data. The first step to graph our data is to compute the correlation matrix representing the different attributes that can be used to predict others using the `corr()` method. This is very useful as it shows what data points can be used to recommend a player and how every player is measured in terms of the attributes presented in our dataset.

This is achieved with the following code:

```python
sample = data.select_dtypes(include='number')
corr = sample.corr()
mask = np.zeros_like(corr, dtype = np.bool_)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10))
sns.heatmap(corr, mask=mask)
```

The graph we came up with in this section is a heatmap that takes the **X** and **Y** axes' attributes and measures correlation in a predetermined scale. The following is our heatmap:



## **Recommendation System** 

In this section, we design and prepare our recommendation system model. For this purpose, we begin by converting our data points to numerical values using a standard scaler. Next, we run the Nearest Neighbors model that will be used as our recommendation system algorithm. The model will take as input our numerical data and will be used to find comparable players that are close in characteristics to our input. 

This is achieved with the following code:

```python
scaled = StandardScaler()
X = scaled.fit_transform(sample)
recommendations = NearestNeighbors(n_neighbors = 5, algorithm='kd_tree')
recommendations.fit(X)
player_index = recommendations.kneighbors(X)[1]
```

We have prepared our model and must create a method to find the player's index in our dataset based on a given name. This is very useful as it will allow us to use the results of the recommendation system to extract the recommended player attributes.

This is achieved with the following code:

```python
def find_index(x):
    return data[data['name']==x].index.tolist()[0]
```

The last step is to create a method using our model to extract the player's information. In other words, we must extract key values describing the recommended players for every player. Our example mainly focuses on the player's name, market value, age, and current club.

This is achieved with the following code:

```python
def recommendation_system(player):
    print("Here are four players who are similar to {}: ".format(player))
    index =  find_index(player)
    
    for i in player_index[index][1:]:
        print("Player Name: {}\nPlayer Market Value: €{}\nPlayer Age: {}\nPlayer Current Club: {}\n".format(
            data.iloc[i]['name'],
            data.iloc[i]['market_value'], 
            data.iloc[i]['age'], 
            data.iloc[i]['club']))
```

## Model Evaluation 

At this moment, we are ready to evaluate our model. In this example, we will run our recommendation system that inputs a player name and recommends **four** key players comparable to the player inputter. 

This code can be achieved as follow:

```python
recommendation_system('Petr Cech')
```

The results of our mode are as follows:

```python
Here are four players who are similar to Petr Cech: 
Player Name: Willy Caballero
Player Market Value: €1.5
Player Age: 35
Player Current Club: Chelsea

Player Name: Nacho Monreal
Player Market Value: €13.0
Player Age: 31
Player Current Club: Arsenal

Player Name: Laurent Koscielny
Player Market Value: €22.0
Player Age: 31
Player Current Club: Arsenal

Player Name: Artur Boruc
Player Market Value: €1.0
Player Age: 37
Player Current Club: Bournemouth
```

As we can see, the recommendation system was able to predict with high accurse five players that are very close to our input players. This proves that our model is ready for deployment and can be used to predict future talent based on a current player.

## **Conclusion** 

Using a recommendation system to create a list of prospects in the business of professional atheism is a competitive advantage any team manager should consider. The article covered the step-by-step process of creating a recommendation model using English Premier League Players. GridDB was extensively used in the article as a database storage for our dataset. This database stores trained data, and it is used to call our recommendation system to recommend players based on their peers.