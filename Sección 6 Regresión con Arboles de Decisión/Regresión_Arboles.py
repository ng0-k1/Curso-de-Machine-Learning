# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:00:29 2021

@author: Usuario
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

y = y.reshape(-1,1)
#Ajustar la regresión del dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state= 0)
regression.fit(x, y)

y_pred = regression.predict([[5.2]])

plt.scatter(x, y, color="red")
plt.plot(x, regression.predict(x), color="blue")
plt.title("Mod regresion De Maquinas de Soporte")
plt.xlabel("Poisición del empleado")
plt.ylabel("Sueldo en dolares(USD)")
plt.show()

