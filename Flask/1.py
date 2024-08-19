# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:28:10 2022

@author: k Sai Shashank
"""
import os
import numpy as np #used for numerical analysis
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
model_knn=pickle.load(open("book.pkl",'rb'))  
us_canada_user_rating_pivot=pd.read_csv("us_canada_user_rating_pivot1.csv")

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
#print(type(query_index))
distances, indices = model_knn.kneighbors(
    us_canada_user_rating_pivot.iloc[query_index:].values.reshape(1, -1), n_neighbors = 6)
print(query_index)