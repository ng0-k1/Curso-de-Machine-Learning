# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 20:01:50 2021

@author: Usuario
"""

import pandas as pd 

dataset = pd.read_csv('Wine.csv')

x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values  

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2 ,random_state=0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

