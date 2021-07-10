#Aprendizaje por refuerzo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')



#Algoritmo de UCB

#cantidad de veces seleccionado  d = num_anuncios
#N = Numero de usuarios
import math
d = 10
N = 10000
number_of_selection = [0]*d
sum_recompensas = [0]*d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range (0,d):
        #Decimos que cuando la selección del numero de selecciones sea al menos 1 entonces empiece a evaluar
        if(number_of_selection[i]>0):
            #Recompensa media
            recompensa_media = sum_recompensas[i]/number_of_selection[i]
            #Calculando el delta, sqrt es raiz
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selection[i])
            #Intervalo de confianza 
            upper_bound = recompensa_media + delta_i
        #De no ser así y el number of selección no sea mayor a 0 entonces tomara el upper como 1 elevado a la 400
        else:
            upper_bound = 1e400
        #El maximo del intervalo de confianza
        if upper_bound >max_upper_bound:
            max_upper_bound = upper_bound
            #Anuncio con mayor indice conocido
            ad = i
    ads_selected.append(ad)
    number_of_selection[ad] = number_of_selection[ad]+1
    recompensa=dataset.values[n, ad]
    sum_recompensas[ad] = sum_recompensas[ad]+recompensa
    total_reward = total_reward+recompensa
    
#Histograma de resultados 
plt.hist(ads_selected)
plt.title("Histogradama de anuncios")
plt.xlabel("Id anuncio")
plt.ylabel("Frecuencia de visualización")
plt.show()

    
        
        
        