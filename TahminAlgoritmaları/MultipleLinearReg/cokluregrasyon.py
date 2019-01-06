# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:58:58 2018

@author: Lenovo
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 ) 
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


c = veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


y_pred=lr.predict(x_test)

# Tüm veriler bakarak boy tahmini yapıyor
#bagımlı değişken
boy=s2.iloc[:,3:4].values
print(boy)

sol =s2.iloc[:,:3]
sag =s2.iloc[:,4:]
#bagımsiz değişkenler
veri=pd.concat([sol,sag],axis=1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


lr2=LinearRegression()
lr2.fit(x_train,y_train)


y_pred2=lr2.predict(x_test)
#Backward Eliminition
import statsmodels.formula.api as sm
 #np.ones 1ler oluşan matris yapar
X= np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l= veri.iloc[:,[0,1,2,3,4,5]].values

r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
print(r.summary())

X_l= veri.iloc[:,[0,1,2,3,5]].values

r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
print(r.summary())


X_l= veri.iloc[:,[0,1,2,3]].values

r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
print(r.summary())


    
    

