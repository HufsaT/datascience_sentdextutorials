# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:10:58 2018

@author: Hufsa
"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val+=step
        elif correlation and correlation == "neg":
            val-=step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    
#plt.scatter(xs,ys)
#plt.show()
def bestfitslope_int(xs,ys):
    m = (((mean(xs)*mean(ys))- mean(xs*ys)) / 
         ((mean(xs)**2) - mean(xs**2)))
    b = (mean(ys)-m*(mean(xs)))
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)
    
def coefficient_det(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    sq_error_reg = squared_error(ys_orig, ys_line)
    sq_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1- (sq_error_reg / sq_error_y_mean)

xs, ys = create_dataset(40,10,2,correlation='pos')
m,b = bestfitslope_int(xs,ys)

#print(m, b)

reg_line = [(m*x) + b for x in xs]

r_sq = coefficient_det(ys, reg_line)
print(r_sq)

plt.scatter(xs,ys)
plt.plot(xs, reg_line)
plt.show()

