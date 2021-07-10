datasett = read.csv('Data.csv')


#Dividir los datos entre conjuntos de entrenamiento y prueba


#Instalando paquetes
#install.packages("caTools")

library(caTools)
#Definir semilla aleatoria
set.seed(123)

#Definimos la división
split = sample.split(datasett$Purchased, SplitRatio = 0.8)
#conjunto de entrenamiento y de test
training_set = subset(datasett, split == TRUE)
testing_set = subset(datasett, split == FALSE)

#Escalado de valores (Es normalizar los datos)
#[fila, columnas:columnas]
# training_set[,2:3] =scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])






