#Preprocesado de datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el dataset

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

#DATOS CATEGORICOS 

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#Creando un codificador de datos
labelencoder_X = preprocessing.LabelEncoder()

#Se encarga de tomar directamente las columnas que indiquemos y transformarlo a datos numericos
#IMPORTANTE, ES DE TENER EN CUENTA QUE LOS DATOS CATEGORICOS SERAN DIFERENTES A LOS ORDINALES
#PUES SE DEBE TENER EN CUENTA QUE LOS CATEGORICOS NO SON COMPARABLES ENTRE SI
x[:, 1] = labelencoder_X.fit_transform(x[:, 1])



labelencoder_X_2 = preprocessing.LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])



#Para crear nuestros dummyes (cuando hacemos uso de esto, nuestros paises ya no se representan solo con 1 2 3 si no en cada columna nos dira ej Francia = 1 0 0 donde 1
#Sera el valor que tomara para el país y 0 sera los paises que no representa)
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[1])],
    remainder = 'passthrough'   
    )
x = np.array(ct.fit_transform(x),dtype=float)
#Quitando una de las filas de la variable dummie para evitar multicolinealidad
x = x[:,1:]

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[3])],
    remainder = 'passthrough'   
    )



#Dividiendo el dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Escalando Variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#CONSTRUYENDO RED NEUROANL 

#Importando Keras
import keras
#Para capas intermedias
from keras.models import Sequential
from keras.layers import Dense

#Inicializar la red neuronal
# = Objeto para iniciar la red
classifier = Sequential()

#Añadir las capas de entrada - Primera capa oculta
#Dense es la conexión entre capas y recibe los parametros de
#(nodos de salida, pesos, función de activación, nodos de entrada )  
#los nodos se seleccionan a veces con la media entre capas de salida y entrada
#el kernel_initializer los inicia con una distribución uniforme
#el relu = rectificador lineal unitario (se puede aplicar otros rectificadores)
#input_dim = nodos de entrada
classifier.add(Dense(units =6, kernel_initializer= "uniform", 
                     activation= "relu", input_dim = 11))

#Segunda Capa Oculta
#Se elimina el input_dim porque ya entiende que llevara la misma de la anterior
classifier.add(Dense(units =6, kernel_initializer= "uniform", 
                     activation= "relu"))

#Capa de salida 
#Cuando se desean varias salidas para el activation
#el units se cambia por la  cantidad de clases que deseamos
#y en el activation se aplicaria por un rectificador unitario o un escalonado
classifier.add(Dense(units = 1, kernel_initializer= "uniform", 
                     activation= "sigmoid"))


#Enganchando los nodos y aplicando procesos estocasticos
#Compilando la red neuronal
#optimizer es el conjunto optimo de pesos, permite elegir un algoritmo para llegar al optimo (desc gradiente, optimizador de adam)
#lost= Función de perdida que permite minimizar el error entre la prediccion y el dato real
#El parametro del loss es aplicar la diferencia y aplicar un algoritmo para convertir de categoria a numerro
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"] )


#AJUSTANDO LA RED NEURONAL AL CONJUNTO DE ENTRENAMIENTO
#epochs = numero de iteraciones globales
#El bathc_size es la cantidad de datos que escogera para 1 iteración
classifier.fit(x_train, y_train, batch_size= 8, epochs= 10)


#PARTE NUMERO 3

y_pred = classifier.predict(x_test)
#filtrando y convirtiendo, todos los valores que esten por encima de 0.5 devolveran como posibles clientes que se van
y_pred = (y_pred>0.5)


#Matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#NOTA A LA HORA DE VER CM LO QUE HAREMOS SERA COMPARAR QUE TANTOS ERRORES TUVIERON DADO LO SIGUIENTE
#|acertada   |No acertada|
#|No acertada|  acertada |

#NOTA el porcentaje de predicción se puede calcular como: acertada+acertada/observaciones (ej (1518+211)/2000 )

