# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:25:43 2021

@author: Usuario
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
#n_estimators son la cantidad de arboles
#criterion = Criterio, para este caso el metodo de minimos cuadrados

regressor = RandomForestRegressor(n_estimators = 200, criterion="mse", random_state=0)
regressor.fit(x, y)




y_pred = regressor.predict([[6.5],[7.5]])

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(x, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Bosques aleatorios ")
plt.xlabel("Poisición del empleado")
plt.ylabel("Sueldo en dolares(USD)")
plt.show()
