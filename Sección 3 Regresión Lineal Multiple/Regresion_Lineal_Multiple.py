"""
Created on Tue Dec 15 22:06:24 2020

@author: Oscar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

 #iloc = localizar elementos por posición y dentro de este le decimos que va hasta -1
 #Esto para que no tome la ultima posición y con el values devolvemos los valores
x = dataset.iloc[:, :-1].values

#Para y es la misma situación pero solo queremos el ultimo elemento            
y = dataset.iloc[:, 4].values

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#Creando un codificador de datos
labelencoder_X = preprocessing.LabelEncoder()

#Se encarga de tomar directamente las columnas que indiquemos y transformarlo a datos numericos
#IMPORTANTE, ES DE TENER EN CUENTA QUE LOS DATOS CATEGORICOS SERAN DIFERENTES A LOS ORDINALES
#PUES SE DEBE TENER EN CUENTA QUE LOS CATEGORICOS NO SON COMPARABLES ENTRE SI
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])

#Para crear nuestros dummyes (cuando hacemos uso de esto, nuestros paises ya no se representan solo con 1 2 3 si no en cada columna nos dira ej Francia = 1 0 0 donde 1
#Sera el valor que tomara para el país y 0 sera los paises que no representa)
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[3])],
    remainder = 'passthrough'   
    )
x = np.array(ct.fit_transform(x),dtype=np.float)

#Evitar la trampa de variables dummyes(Eliminar siempre una variable)
x = x[:, 1:]


#Dividir el dataset en el entrenamiento y en el testing
#train_test_split es la función que permite dividir el dataset
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)



#Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
#Diciendole que ajustaremos que en este caso es el conjunto de entrenamiento
#IMPORTANTE al dejar el x_train sin ninguna especificación ha asumido todas las variables que lleva consigo (en este caso son 5)
regression.fit(x_train, y_train)

#Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(x_test)

#--------------------------------------------
#(Construytendo el modelo optimo de RLM  usando )ELIMINACIÓN HACIA ATRAS
#Eliminando aquellas variables que no tienen un mayor impacto o que no son estadisticamente significativas del modelo para mejorar la predicción
#IMPORTANTE, YA NO SE HACE USO DE FORMULA.API SI NO STATSMODELS.API
import statsmodels.api as sm
#Reasignamos a traves del statsmodel(libreria para aplicar eliminación hacia atras)
#a traves de reasignar x = le diremos que nos agregue un array donde le pasaremos primero el parametro de np.ones para que en la primer columna coja el valor 
#de 1 y dentro del metodo le diremos 50,1 para decirle que son 50 valores asignados a 1 columna, además de eso parseamos los valores a entero (por defecto esta en float), le diremos luego el valor y la posición donde lo ubicaremos
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
SL = 0.05 #Nivel de significación

x_opt = x[:, [0,1,2,3,4,5]].tolist()#Variable que se encarga de guardar el numero optimo de las variables independientes
#OLS = Ordinary List Square (metodo de los minimos cuadrados)
regression_OLS = sm.OLS(y,x_opt).fit()  #endog (endogena) variable a predecir - Exog(exogena) variable independiente
#Sumando los valores para tener los valores estadisticos (se mira mayormente los valores de los coeficientes y )
regression_OLS.summary()

x_opt = x[:, [0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(y,x_opt).fit()
regression_OLS.summary()

x_opt = x[:, [0,3,4,5]].tolist()
regression_OLS = sm.OLS(y,x_opt).fit()
regression_OLS.summary()

x_opt = x[:, [0,3]].tolist()
regression_OLS = sm.OLS(y,x_opt).fit()
regression_OLS.summary()

import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



