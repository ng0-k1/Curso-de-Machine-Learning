# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:49:03 2021

@author: Usuario
"""

import pandas as pd
import numpy as np 

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracie = cross_val_score(estimator= classifier, X= x_train, y= y_train, cv=10)
accuracie.mean()
accuracie.std()

#Aplicando la mejora de grid search para mejorar sus parametros 
from sklearn.model_selection import GridSearchCV
#Lista de python con los parametros que se usan
parameters =[{'C':[1,10,100,1000],'kernel': ['linear']},
             {'C':[1,10,100,1000],'kernel': ['rbf'], 'gamma':[0.5,0.6,0.7,0.001]}
             ]
#Pasamos el parametro estimador, asignamos un score (para este caso el accuracy)
#Creamos un cross validation y asignamos el nmero de cores que escogera todos menos el del S.O
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)

#Mejor predicción
best_accuracy = grid_search.best_score_
#Mejores parametros
best_parameters = grid_search.best_params_


