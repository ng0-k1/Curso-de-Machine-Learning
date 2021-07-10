# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:31:45 2020

@author: Usuario
"""

#Plantilla de Pre procesado - Datos categoricos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

 #iloc = localizar elementos por posición y dentro de este le decimos que va hasta -1
 #Esto para que no tome la ultima posición y con el values devolvemos los valores
 
x = dataset.iloc[:, :-1].values

#Para y es la misma situación pero solo queremos el ultimo elemento            
y = dataset.iloc[:, 3].values


#CODIFICAR DATOS CATEGORICOS, toca importar OneHotEncoder para crear datos dummy y los diferencia como categoricos

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#Creando un codificador de datos
labelencoder_X = preprocessing.LabelEncoder()

#Se encarga de tomar directamente las columnas que indiquemos y transformarlo a datos numericos
#IMPORTANTE, ES DE TENER EN CUENTA QUE LOS DATOS CATEGORICOS SERAN DIFERENTES A LOS ORDINALES
#PUES SE DEBE TENER EN CUENTA QUE LOS CATEGORICOS NO SON COMPARABLES ENTRE SI
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])

#Para crear nuestros dummyes (cuando hacemos uso de esto, nuestros paises ya no se representan solo con 1 2 3 si no en cada columna nos dira ej Francia = 1 0 0 donde 1
#Sera el valor que tomara para el país y 0 sera los paises que no representa)
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[0])],
    remainder = 'passthrough'   
    )
x = np.array(ct.fit_transform(x),dtype=np.float)

#Para transormar la columna de Purchase
lbl_Y = preprocessing.LabelEncoder()
y = lbl_Y.fit_transform(y)