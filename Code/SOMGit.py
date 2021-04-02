#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *

#Dataset is consisted with an index column
dataset=pd.read_csv("/pathname/Dataset NameInd.csv", sep=',',header=None) 
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.shape


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 9, sigma = 1.0, learning_rate = 0.3)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar
bone()

distance_map = som.distance_map().round(1)

index = []
for i in range(10):
    for j in range(10):
        if(distance_map[i,j]>=0.5):
            index.append([i,j])
len(index)

pcolor(som.distance_map().T)
colorbar() #gives legend

mappings = som.win_map(X)
mappings.keys()

outlier_list = []
sum = 0
for x in index:
    outlier_list.append(mappings[(x[0],x[1])])
    sum = sum + len(mappings[(x[0],x[1])])
sum

outliers = []
for x in outlier_list:
    for y in x:
        outliers.append(y)

outlier_array = np.asarray(outliers)

outlier_inverse_transformed = sc.inverse_transform(outliers) # Undo the scaling of data pattern of frauds according to feature_range

count = 0
outlier_id_list = []
for x in outlier_inverse_transformed:
    outlier_id_list.append(x[0])
print('Total outliers :{}'.format(len(outlier_id_list)))

X_index=dataset.iloc[:, [0]].values
X_index_flat=X_index.flatten()
X_index_trans=X_index_flat.tolist()
common_item=set(outlier_id_list).intersection(X_index_trans)
indices=[X_index_trans.index(ind) for ind in common_item]
indices.sort()
print(indices)
