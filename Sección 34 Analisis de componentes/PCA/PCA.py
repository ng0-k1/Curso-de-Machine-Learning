# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:29:44 2021

@author: Usuario
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('Wine.csv')

x = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)

#Reducir la dimensión del dataset con ACP

from sklearn.decomposition import PCA
#n_components = cantidad de variables que deseamos
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#Componentes principales y porcentajes
explained_variance = pca.explained_variance_ratio_

#Regresión Logistica
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state=4)
regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

