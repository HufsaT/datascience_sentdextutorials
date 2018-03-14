# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:22:49 2018

@author: Hufsa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 00:25:02 2018

@author: Hufsa
"""
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1 ,inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = svm.SVC()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)


        

        