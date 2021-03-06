#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import SOMEnsemble as smE
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv("/pathname/Dataset Name.csv", sep=',',header=None)
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)



X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values


NoiseInd=smE.outlier_common_misclf
X1=np.delete(X,NoiseInd,axis=0)

y1=np.delete(y,NoiseInd,axis=0)



X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3,random_state=109)




clfJ48N=DecisionTreeClassifier()
clfJ48N.fit(X_train, y_train)
y_pred_J48N=clfJ48N.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_J48N))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb=gnb.predict(X_test)
print("Accuracy NB after noise remove:",metrics.accuracy_score(y_test, y_pred_nb))

Knn = KNeighborsClassifier(n_neighbors=3)
Knn.fit(X_train, y_train)
y_pred_Knn=Knn.predict(X_test)
#print(" KNN Accuracy:",metrics.accuracy_score(y_test, y_pred_Knn))

RF = RandomForestClassifier(max_depth=2, random_state=0)
RF.fit(X_train, y_train)
y_pred_RF=RF.predict(X_test)
#print(" RF Accuracy:",metrics.accuracy_score(y_test, y_pred_RF))

SV=svm.SVC()
SV.fit(X_train, y_train)
y_pred_SV=SV.predict(X_test)
print(" SVM Accuracy:",metrics.accuracy_score(y_test, y_pred_SV))
