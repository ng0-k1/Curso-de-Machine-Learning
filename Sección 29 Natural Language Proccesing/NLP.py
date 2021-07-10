# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:41:43 2021

@author: Usuario
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#Quoting es para ignorar y en este caso el 3 es comillas dobles
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter= "\t", quoting= 3)
