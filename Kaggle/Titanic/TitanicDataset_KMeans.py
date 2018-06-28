# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:21:25 2018

@author: Hufsa
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')

df = pd.read_excel("titanic.xls")
df.drop(['body','name','ticket','embarked','home.dest'], 1, inplace=True)
df.fillna(0,inplace=True)

# using dummies without dropping a column to give each category variable a number
new_df = pd.get_dummies(df, columns=['pclass','sex','sibsp','parch','cabin','boat'])

#print(df.head())
#print(new_df.head())

# setting all the new binary values to a float just in case
# dropping survived col so we cna test against it later
x = np.array(new_df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(new_df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x,y)

correct = 0

for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
#    print(predict_me)
    prediction = clf.predict(predict_me)
#    print(prediction)
    if prediction[0] == y[i]:
        correct+=1
        
print(correct/len(x))
