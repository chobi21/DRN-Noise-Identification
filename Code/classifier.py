# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

dataset=pd.read_csv("/pathname/Dataset Name.csv", sep=',',header=None)
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0, 1))
#X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)



clfJ48=DecisionTreeClassifier()
clfJ48.fit(X_train, y_train)
y_pred_J48=clfJ48.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_J48))

misclassified_J48 = np.where(y_test != y_pred_J48)

mTupELen=len(misclassified_J48[-1])

msClsList=[]

for x in range(mTupELen):
    a=misclassified_J48[-1][x]
    msClsList.append(a)
#print(msClsList)


#print(misclassified_J48)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb=gnb.predict(X_test)
print("Accuracy NB:",metrics.accuracy_score(y_test, y_pred_nb))
misclassified_nb = np.where(y_test != y_pred_nb)
mTupELenNB=len(misclassified_nb[-1])
msClsListNB=[]

for p in range(mTupELenNB):
    ap=misclassified_nb[-1][p]
    msClsListNB.append(ap)