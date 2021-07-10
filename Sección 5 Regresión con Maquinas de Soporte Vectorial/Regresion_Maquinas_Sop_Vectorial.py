"""
Created on Wed Feb 17 18:26:42 2021

@author: Oscar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

 #iloc = localizar elementos por posición y dentro de este le decimos que va hasta -1
 #Esto para que no tome la ultima posición y con el values devolvemos los valores
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Escalando variables
from sklearn.preprocessing import StandardScaler
#Diciendole entre que valores debe escalar
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
#Dentro del SVR, seleccionamos el kernel que en este caso es un gaussiano
regression = SVR(kernel="rbf")
regression.fit(x, y)

#Mirando la predicción del modelo
y_pred =sc_y.inverse_transform(regression.predict(sc_x.transform([[6.5]])))


#Mostramos x y dentro de y lo que predeciremos
plt.scatter(x, y, color="red")
plt.plot(x, regression.predict(x), color="blue")
plt.title("Mod regresion De Maquinas de Soporte")
plt.xlabel("Poisición del empleado")
plt.ylabel("Sueldo en dolares(USD)")
plt.show()







