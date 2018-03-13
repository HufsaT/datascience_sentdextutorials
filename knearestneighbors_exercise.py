# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 00:25:02 2018

@author: Hufsa
"""
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest(data, predict, k=3):
    if len(data)>=3:
        warnings.warn('K value is less than or equal to groupings.')
    distances = []
    for group in data:
        for features in data[group]:
            eu_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([eu_distance, group])
        
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0] # bc this will normally output a tuple with list so we grab the first position (the list) and then the first item, class name only
    return vote_result


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1 ,inplace=True)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data) # do not need to reset full_data to this.
test_size = 0.2
train_set ={2:[], 4:[]} # dicts for classifications
test_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size*len(full_data))] # everything up to 20% of data is training
test_data = full_data[-int(test_size*len(full_data)):] # last 20% of set is testing

# now we have ot sort the traina dn test lists into the right dicts. We will use the last list element in each sample to help classify if it is a 2 or a 4 class.

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# now for each piece of test sample, we check it against the training data to find its nearest neighbors
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest(train_set, data, k=5)
        if group == vote:
            correct+=1
        total +=1
print('Accuracy:', correct/total)   
        

        