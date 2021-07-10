# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:31:49 2020

@author: Usuario
"""

#Plantilla de Pre  - Datos Faltantes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

 #iloc = localizar elementos por posición y dentro de este le decimos que va hasta -1
 #Esto para que no tome la ultima posición y con el values devolvemos los valores
 
x = dataset.iloc[:, :-1].values

#Para y es la misma situación pero solo queremos el ultimo elemento            
y = dataset.iloc[:, 3].values



#La libreria contiene metodos para el pre procesado de datos 
# IMPORTANTE el apartado Imputer esta en desuso, ahora se usa SimpleImputer y ahora se llama a la libreria como sklearn.imputer y no prepocessing
from sklearn.impute import SimpleImputer

#Valores detectados como desconocidos diciendole que buscara los 
#Con strategy le diremos que estrategia usara para reemplazar los nan (mean = media)
#El axis lo usaremos para indicar si sustituiremos el valor de la fila o columna si necesitamos 
#IMPORTANTE estan en desuso el metodo axis (la fila le diremos axis = 1 si no le diremos 0)
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

#fit = aplicar una función a un objeto y le diremos que tome las columnas de age y salary
# le diremos que tome entonces 2 y 3
imputer = imputer.fit(x[:, 1:3])

#Aquí transformaremos la variable x 
x[:, 1:3] = imputer.transform(x[:,1:3])




