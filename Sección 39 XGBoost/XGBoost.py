# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:34:51 2021

@author: Usuario
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')


x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
labelencoder_X = preprocessing.LabelEncoder()
x[:, 1] = labelencoder_X.fit_transform(x[:, 1])

labelencoder_X_2 = preprocessing.LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])

#Para crear nuestros dummyes (cuando hacemos uso de esto, nuestros paises ya no se representan solo con 1 2 3 si no en cada columna nos dira ej Francia = 1 0 0 donde 1
#Sera el valor que tomara para el país y 0 sera los paises que no representa)
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[1])],
    remainder = 'passthrough'   
    )
x = np.array(ct.fit_transform(x),dtype=float)
#Quitando una de las filas de la variable dummie para evitar multicolinealidad
x = x[:,1:]

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[3])],
    remainder = 'passthrough'   
    )



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Ajustar XGBoost al conjunto de entrenamiento 
#Algoritmo usado para clasificar 
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracie = cross_val_score(estimator= classifier, X= x_train, y= y_train, cv=15)
por = accuracie.mean()