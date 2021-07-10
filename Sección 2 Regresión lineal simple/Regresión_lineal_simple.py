# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:09:22 2020

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
#Independiente, tomara todos los valores menos la ultima
x = dataset.iloc[:, :-1].values
#Dependiente            
y = dataset.iloc[:, 1].values

#División de variables para prueba y entrenamiento
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=0)

#Cuando es una regresión lineal simple no se hace necesario escalar 
#los datos
#Escalado de variables
#crear un escalador estandar
'''
from sklearn.preprocessing import StandardScaler

#Diciendole entre que valores debe escalar
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

#Escalando el test y no se debe volver a crear un escalador (sc_x)
#Y solo le diremos transform para que escale con el test 
x_test = sc_x.transform(x_test)
'''
#--------------------------------------------
#Crear modelo de regresión lineal simple con el conjunto de entrenamiento

#Importando la libreria para crear modelos de regresión lineal
from sklearn.linear_model import LinearRegression
#invocando la función donde llamaremos la regresión
regression = LinearRegression()
#Ajustar el modelo usando el modelo de la clase (debe tener mismo numero de filas tanto x como y)
regression.fit(x_train, y_train)
#----------------------
#Predecir el conjunto de test
#Es de aclarar que ya se le ha suministrado los conjuntos de entrenamiento

#Creando un vector de predicciones, se debe tomar solo los valores independientes
y_pred = regression.predict(x_test)


#Visualizando los datos con matplotlib de los entrenamientos
#En el eje X tendremos años y en los Y los salarios, esto es a traves de nubes de dispersión
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regression.predict(x_train), color="blue")
plt.title("Sueldo en relación con los años del conjunto de entrenamiento")
plt.xlabel("Años de experiencia")
plt.ylabel("Salarios")
plt.show()

#PROBANDO EL CONJUNTO DE TEST(ES IMPORTANTE SABER QUE ESTOS VALORES
#DEBEN SER SIMILARES A LOS DE ENTRENAMIENTO O SI NO ESTOS TEST NO SERVIRAN)
#Visualizando los datos con matplotlib de los test
#En el eje X tendremos años y en los Y los salarios, esto es a traves de nubes de dispersión
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regression.predict(x_train), color="blue")
plt.title("Sueldo en relación con los años del conjunto de test")
plt.xlabel("Años de experiencia")
plt.ylabel("Salarios")
plt.show()




