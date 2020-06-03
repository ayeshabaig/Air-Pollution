import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import ResidualsPlot
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import GridSearchCV
import pickle

from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import classification_report






# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.

df = pd.read_csv("bigtable.csv").dropna()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


map_load_state = st.text('Loading map...')
df['lat'] = df['x']
df['lon'] = df['y']
st.map(df)
map_load_state.text('Loading map...done!')


process_cols = [
    ['x', 'y', 'population', 'dist-mroads',
       'dist-setl', 'dist-coast', 'dist-forest', 'slope', 'elevation',
       'dayofweek', 'sin_day', 'cos_day', 'sin_year', 'cos_year', 'TEMP',
       'DEW', 'SKY', 'VIS', 'ATM', 'Wind-Rate', 'sin_wind', 'cos_wind']
]
for cols in process_cols:
    # select the target variable
    y = df['pm25']
    # select the selected columns for the iteration
    X = df[cols] 




X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, y_test)
#st.write("The score of the loaded model is :",result)




#X_test = X_test[np.logical_and(X_test["x"] == x1, X_test["y"] == y1)]
user_input = int(st.text_input(f"Select values from 0 to {len(X_test)-1}", 0))
to_pred = X_test.iloc[user_input:user_input+1,:]

def rmse_predict(val1, val2):
    return (val1**2 - val2**2)**(1/2)
st.write("the selected data is")
st.write(to_pred)
val1 = loaded_model.predict(to_pred)[0]
val2 = y_test.values[user_input]
st.write(f"the actual pm25 val is {val2}")
rmse = rmse_predict(val1, val2)
st.write(f"The predicted value of pm25 is {val1}")
st.write(f"the rmse for predicted value is {(abs(val1**2 - val2**2))**(1/2)}")



st.write("""
#The analysis dashboard
""")
st.write("the first 5 data sample")
st.write(df.head())
dfs = df.describe()
#the visualization
namez = [] ## create empty list 
for x in dfs.describe().columns: ## get column names
    namez.append(x)      ### append to a list namez 

ys = dfs.describe().values[1:] ### get all arrays of values except for the first which is count
x1 = dfs.describe().columns[3:]  
texts = ['count','mean', 'std', 'min', '25%', '50%', '75%', 'max'] ### we will use these labels to show information when hovering on the chart
fig = go.Figure() ### create an empty figure so that we can then append some data into it
for ll in range(0,len(texts)):
    fig.add_trace((go.Bar(x=x1,y=ys[ll-1], name=texts[ll],text=texts[ll],hoverinfo='text+y'))) ## add a new trace on hte graph for each of the columns 
fig.update_layout(barmode='group',title="Air Pollution <br> Using Decribe Data") ## call barmode as group
st.write("the data description plot")
st.write(df.describe())
st.write(fig)