#Plantilla de Pre procesado

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000

d = 10
#numero de veces que uno recibio el premio
number_of_rewards_1 = [0]*d
#numero de veces que no nos da recompensas
number_of_rewards_0 = [0]*d
ads_selected = []
total_reward = 0

for n in range (0, N):
    #Seleccionar el maximo de los valores aleatorios
    max_random = 0
    ad = 0
    for i in range (0,d):
        #Un valor random
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward ==1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad]+1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[0]+1
    total_reward = total_reward+reward


plt.hist(ads_selected)
plt.title("Prueba de Thompson")
plt.xlabel("Id_anuncio")
plt.ylabel("Frecuencia de visualización")