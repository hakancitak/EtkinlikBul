# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:40:49 2019

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

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,rf_reg.predict(x.values),color='blue')
plt.show()

from sklearn.metrics import r2_score
print("Random Forest R2 Değeri= ")
print(r2_score(y.values,rf_reg.predict(x.values)))