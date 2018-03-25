# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:48:59 2018

@author: Hufsa
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    
    def __init__(self,visualization=True): # do we want to visualize all data?
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train       
    def fit(self,data):
        self.data = data
        # mag of W is key, vector w and b are values
        opt_dict = {}
        transforms = [[1,1],[-1,-1],[1,-1],[-1,1]]
        
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # we basically need to find a value for yi(w*x + b) = 1
        
        # taking bigger steps and then fine-tuning to find the best minimized w vector
        step_size = [self.max_feature_value * 0.1,
                     self.max_feature_value * 0.01,
                     # point of expense
                     self.max_feature_value * 0.001]
        
        # ext expensive to find max b
        b_range_multiple = 2 # taking bigger steps for b
        b_multiple = 5
        
        # first step, sert vec W to ten x bigger than max val in data
        latest_optimum = self.max_feature_value * 10
        
        for step in step_size:
            # set both w and b to max_val * 10
            w = np.array([latest_optimum,latest_optimum])
            # e.g. 8 being max, w = 80, 80
            # we keep opt as false till we find lowest magnitude. Since convex problem, we will only have one absolute min
            optimized = False
            
            while not optimized:
                # while looking for smallest w we look for largest b
                # range of neg max val * step to pos max val* step in dataset, step in range is set to 5
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    # e.g. b = from (-40 to + 40) stepping 40
                    # e.g. b is -40, w is 80,80
                    
                    # we are not treating b same as w, so not taking smaller and smaller steps b/w possible values.
                    # but we could
                    for transformation in transforms:
                        w_t = w*transformation 
                        # e.g. w = 80,80
                        found_option = True
                        
#                        # weakest link in SVM, must run this calc on all data to see if fits
                        for i in self.data:
                            for xi in self.data[i]:
                                # yi(xi * w + b)>= 1 is our constraint so our w and b must satisfy this
                                # we insist that positive samples be >= 1 and neg be <= -1
                                
                                yi = i # just for clarity
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                        if found_option:
                            # now add each w and b value with its magnitude as a key to a new dict
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    # because if w goes below 0, norm/dot prod will be 5 (from b)
                    print('Optimized a step!')
                else:
                    w = w - step # reduce w to next step and continue main loop again
                    
            # now sort the optimized values for smallest magnitude
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]] # will return mag-w: w,b
            self.w = opt_choice[0] # get w as a list item
            self.b = opt_choice[1] # get b as a single value
            # [0][0] is because w is a list, you only need first w to be changed in next loop iteration
            latest_optimum = opt_choice[0][0]+step * 10 # set starting w to current lowest value of w
        
    # return the sign pos or neg of the final answer    
                
    def predict(self,features):
        # sign of (x * w + b), will return 1 or -1
        # here you are reflecting the unknown feature set onto vector w and adding b to se where it lands.
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        # now graphing this new classified item
        if classification != 0 and self.visualization:
            # if user wants us to visualize, we do this.
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        else:
            print('featureset',features,'is on the decision boundary')
        return classification
        
    def visualize(self):
        # plots existing points in data set (not unknown points)
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in self.data[i]] for i in self.data]
      
        
        # plots hyperlane just for us to see, not part of svm
            # v = xw + b
            # pos, v = 1
            # neg, v = -1
            # x_v * w + b = 1 for a pos class
            # here 1 subs in for v y-value
            # [x,y] * [w0,w1] + b = 1
            # xw0 + yw1 + b = 1
            # solve for y coordinates for pos, middle, and neg 
            # so we know where hyperplan should lie 
            # y = (1-b-xw0) / w1 
            # this class will be called later with a specific x-coordinate
            
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
        
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')
        
         # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()
        
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)
    
svm.visualize()
