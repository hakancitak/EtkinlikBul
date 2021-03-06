# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:12:05 2019

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:20:14 2019

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

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

svc = SVC(kernel='rbf')#linear,poly,rbf
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)