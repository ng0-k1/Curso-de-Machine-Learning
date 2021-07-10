# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:37:17 2021

@author: Usuario
"""

import pandas as pd 
import numpy as np

dataset = pd.read_csv('Social_Network_Ads.csv')

#Columnas a elegir
x = dataset.iloc[:,2:3].values
y = dataset.iloc[:,4].values

#División del dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 0)

#Escalado
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


from sklearn.decomposition import KernelPCA
#Funcion radeal = rbf
k_pca = KernelPCA(n_components=2, kernel='rbf')
x_train = k_pca.fit_transform(x_train)
x_test = k_pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)