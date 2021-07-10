#Regresión lineal simple
datasett = read.csv('Salary_data.csv')


#Dividir los datos entre conjuntos de entrenamiento y prueba


#Instalando paquetes
#install.packages("caTools")

library(caTools)
#Definir semilla aleatoria
set.seed(123)

#Definimos la división
split = sample.split(datasett$Salary, SplitRatio = 2/3)
#conjunto de entrenamiento y de test
training_set = subset(datasett, split == TRUE)
testing_set = subset(datasett, split == FALSE)

#Escalado de valores (Es normalizar los datos)
#[fila, columnas:columnas]
# training_set[,2:3] =scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

#Ajustar el modelo de regresión linela simple con el conjunto de entrenamiento
#Escribiendo la formula de la combinación lineal de como la variable independiente va a estar en función de la dependiente
# ~ = en función a (se pone primero la variable dependiente de la independiente)
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Hacer un resumen de la lista (Es importante tener en cuenta que el P valor (Pr(>|t|)) para cuando se tiene más variables independientes)
#Cuando se muestren más *** más importante es 
#Cuando se deba elegir entre varias variabels el R-squared más alto sera la mejor opción
summary(regressor)


#Predecir el resultado con el conjunto de test
#Para este metodo se debe tener mismo nombre de columnas 
#Se le indica el modelo de predicción y el conjunto de test
y_pred = predict(regressor, newdata = testing_set)

#Visualizando los datos del conjunto de entrenamiento 
#con el + agregamos otro elemento
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), #Nubes de dispersión, el aes es para la estetica
             colour = "red") + 
  #En la geometria de la linea le diremos cual sera X y cual sera Y y dentro de Y mostraremos los pronosticos
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata =training_set)),
            colour = "blue")+
  ggtitle("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")+
  xlab("Años de experiencia")+
  ylab("Sueldo")


#Visualizando los datos del conjunto de test
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary), #Nubes de dispersión, el aes es para la estetica
             colour = "red") + 
  #En la geometria de la linea le diremos cual sera X y cual sera Y, es importante entender que se toma las variables de entrenamiento
  #Pues esta sera nuestra linea que determina un pronostico general
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata =training_set)),
            colour = "blue")+
  ggtitle("Sueldo vs Años de experiencia (Conjunto de test)")+
  xlab("Años de experiencia")+
  ylab("Sueldo")