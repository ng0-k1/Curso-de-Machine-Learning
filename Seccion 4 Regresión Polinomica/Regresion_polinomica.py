import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
#Se debe poner 1:2 para que se lea como una matriz y no un vector
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#No se debe dividir los datos por la cantidad tan poca de datos que se tiene 
#Además de que para el problema planteado no tiene sentido dividir los datos

#Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#Visualización de los resultados lineales
plt.scatter(x, y, color="red")
#Mostramos x y dentro de y lo que predeciremos
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Mod regresiín lineal")
plt.xlabel("Poisición del empleado")
plt.ylabel("Sueldo en dolares(USD)")
plt.show()

#Ajustar la regresión polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 6)
#El fit_transform transforma la variable anterior a una nueva variable
x_poly = pol_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualización de los resultados polinomicos
plt.scatter(x, y, color="red")
#Mostramos x y dentro de y lo que predeciremos
plt.plot(x, lin_reg2.predict(x_poly), color="blue")
plt.title("Mod regresion polinomica")
plt.xlabel("Poisición del empleado")
plt.ylabel("Sueldo en dolares(USD)")
plt.show()

#-----
#Predicción de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg2.predict(pol_reg.fit_transform([[6.5]]))