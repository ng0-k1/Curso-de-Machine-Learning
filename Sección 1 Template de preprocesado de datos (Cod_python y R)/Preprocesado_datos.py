#Plantilla de Pre procesado

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

 #iloc = localizar elementos por posición y dentro de este le decimos que va hasta -1
 #Esto para que no tome la ultima posición y con el values devolvemos los valores
x = dataset.iloc[:, :-1].values

#Para y es la misma situación pero solo queremos el ultimo elemento            
y = dataset.iloc[:, 3].values

#Dividir el dataset en el entrenamiento y en el testing
#train_test_split es la función que permite dividir el dataset
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


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














