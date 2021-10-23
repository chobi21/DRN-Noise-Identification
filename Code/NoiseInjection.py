#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

dataset=pd.read_csv("/pathname/Dataset Name.csv", sep=',',header=None)
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


mean=0
var=0.1
sigma=var**0.5

X1=np.random.normal(mean,sigma, X.shape)
X2=X+X1
data=np.column_stack([X2,y])

np.savetxt("/pathname/DatasetNameNoisy.csv", data,delimiter=",")
