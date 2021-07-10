# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:08:06 2021

@author: Usuario
"""

#K - Fold Cross Validation
import pandas as pd
import numpy as np

dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, 4].values

#Dividiendo en test y entrenamiento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2, random_state=0)

#Escalador de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Modelo SVM
from sklearn.svm import SVC
classifier = SVC(kernel ="rbf", random_state=0)
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

#matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

#Aplicando k fold cross validation

#se aplica al conjunto de entrenamiento y devuelve predicciones del algoritmo
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 15)
#promedio
accuracies.mean()
#Desviación estandar
accuracies.std()  #Para obtener el promedio de aceptación lo que se puede hacer es el accuracie-desviación estandar
#Un ejemplo seria  accuracie = 0.90, desviación = 0.05 entonces el modelo tendria variaciones entre 0.90-0.05 y 0.90+0.05

