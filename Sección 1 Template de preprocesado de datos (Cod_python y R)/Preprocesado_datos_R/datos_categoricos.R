#Datos Categoricos - Preprocesado de datos
datasett = read.csv('Data.csv')

#Codificar las variables categoricas y dentro de los levels asignamos los valores que tenemos
#Y en los labels son los valores que vamos a reemplazar dentro de nuestro dataset
datasett$Country = factor(datasett$Country,
                          levels = c("France","Spain","Germany"),
                          labels = c(1,2,3)
)
datasett$Purchased = factor(datasett$Purchased,
                            levels = c("Yes","No"),
                            labels = c(1,0)
)
