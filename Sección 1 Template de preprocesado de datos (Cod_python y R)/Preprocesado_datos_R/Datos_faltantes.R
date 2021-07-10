# Datos Faltantes - Preprocesado
datasett = read.csv('Data.csv')

#Limpiando datos en R, y para acceder a la variable se usa un $
#Para decirle a los valores que tomen la media lo que haremos sera crear una variable
#(para este caso es ave), dentro de esta le dire que sera una función que recibe el 
#parametro de x  y a este le sacara la media (mean) sin incluir los na (na = Verdadero)
datasett$Age = ifelse(is.na(datasett$Age),
                      ave(datasett$Age, FUN = function(x) mean (x, na.rm = TRUE)),
                      datasett$Age)


datasett$Salary = ifelse(is.na(datasett$Salary),
                         ave(datasett$Salary, FUN = function(x) mean (x, na.rm = TRUE)),
                         datasett$Salary)

