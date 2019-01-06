# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:14:27 2019

@author: Lenovo
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.1. Veri Yukleme
veriler = pd.read_csv('maaslar.csv')
#pd.read_csv("veriler.csv")
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,dt_reg.predict(x.values),color='blue')
plt.show()

from sklearn.metrics import r2_score
print("Decision Tree R2 Değeri= ")
print(r2_score(y.values,dt_reg.predict(x.values)))