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

#dataframe dilimleme
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.linear_model import LinearRegression
#polinomal regresyon 4.dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x.values)
#print(x_poly)
lin_regtwo=LinearRegression()
lin_regtwo.fit(x_poly,y)

plt.scatter(x.values,y.values,color='red')
plt.plot(x.values,lin_regtwo.predict(poly_reg.fit_transform(x.values)),color='blue')
plt.show()
    
    

