dataset = read.csv('50_startups.csv')

#Datos Categoricos
#Codificar las variables categoricas 
dataset$State = factor(dataset$State,
                       levels = c("New York","California","Florida"),
                       labels = c(1,2,3)
)

#Dividir los datos entre conjuntos de entrenamiento y prueba
#Instalando paquetes
#install.packages("caTools")

library(caTools)
#Definir semilla aleatoria
set.seed(123)

#Definimos la división (COLUMNA QUE HAREMOS LA PREDICCION (SAMPLE.SPLIT))
split = sample.split(dataset$Profit, SplitRatio = 0.8)
#conjunto de entrenamiento y de test
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

#Ajustar el modelo de regresioón lineal multiple con el conjunto de entrenamiento
#Con el . le diremos que creara el modelo en función de las demas variables
regressor = lm(formula = Profit ~ .,
               data = training_set)
#IMPORTANTE, EL SUMMARY AUTOMATICAMENTE CREA LA VARIABLE DUMMY Y 
#NO SE DEBE ELIMINAR NADA, ESTE LO HACE AUTOMATICAMENTE
summary(regressor)

#Predecir los resultados con el conjunto de testing
y_pred = predict(regressor, newdata = testing_set)


#ELIMINACIÓN HACIA ATRAS

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

