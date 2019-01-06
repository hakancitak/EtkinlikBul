# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:24:35 2019

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



#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcek = sc1.fit_transform(x.values)
sc2 = StandardScaler()
y_olcek = sc2.fit_transform(y.values)

from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcek,y_olcek)

plt.scatter(x_olcek,y_olcek,color='red')
plt.plot(x_olcek,svr_reg.predict(x_olcek),color='blue')
    

