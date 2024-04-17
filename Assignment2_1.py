# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:21:10 2024

@author: 86130
"""

import pandas as pd

import numpy as np

# Question 1--------------------

ID = list(range(1,11))

Height = [152,180,168,175,142,173,160,173,165,167]

Weight = [77,47,55,59,72,62,40,60,45,58]

Age = [45,26,30,34,40,36,19,28,23,32]

Class = [3,1,2,2,3,2,1,2,1,2]

train_data = pd.DataFrame({'ID':ID,'Height':Height,'Weight':Weight,'Age':Age,'Class':Class})

def EuclideanDistance4KNN(target_data,train_data,K):
    x = target_data[0]
    y = target_data[1]
    X = []
    for i in range(len(train_data)):
        X.append(((train_data['Height'].values[i] - x)**2+(train_data['Weight'].values[i] - y)**2)**0.5)
    train_data['Distance'] = X
    ans = train_data.sort_values(by = 'Distance')
    return round(np.mean(ans['Class'].values[0:K])),round(np.mean(ans['Age'].values[0:K]))

def KMEAN(target_data,counter,c1,c2,c3):
    target_data['c1'] = ((target_data['Height'].values - c1[0])**2 + (target_data['Weight'].values-c1[1])**2)**0.5
    target_data['c2'] = ((target_data['Height'].values - c2[0])**2 + (target_data['Weight'].values-c2[1])**2)**0.5
    target_data['c3'] = ((target_data['Height'].values - c3[0])**2 + (target_data['Weight'].values-c3[1])**2)**0.5
    target_data['Class_new'] = target_data[['c1','c2','c3']].idxmin(1)
    c1_temp = target_data[target_data['Class_new']=='c1']
    c2_temp = target_data[target_data['Class_new']=='c2']
    c3_temp = target_data[target_data['Class_new']=='c3']
    c1 = [np.mean(c1_temp['Height'].values),np.mean(c1_temp['Weight'].values)]
    c2 = [np.mean(c2_temp['Height'].values),np.mean(c2_temp['Weight'].values)]
    c3 = [np.mean(c3_temp['Height'].values),np.mean(c3_temp['Weight'].values)]
    return target_data,c1,c2,c3
    

target_data = [165,50]

ans,age = EuclideanDistance4KNN(target_data,train_data,3)
counter = 0
c1_last = [142,75]
c2_last = [160,45]
c3_last = [173,60]
train_data,c1_new,c2_new,c3_new = KMEAN(train_data,counter,c1_last,c2_last,c3_last)
counter = counter+1

while [c1_last, c2_last,c3_last]!=[c1_new,c2_new,c3_new]:
    c1_last = c1_new
    c2_last = c2_new
    c3_last = c3_new
    train_data,c1_new,c2_new,c3_new = KMEAN(train_data,counter,c1_last,c2_last,c3_last)
    counter = counter + 1
    print(counter)

#Split into two cluster

X_split = train_data.sort_values(by = 'Height')
Y_split = train_data.sort_values(by='Weight')
X_split.index = list(range(len(X_split)))
Y_split.index = list(range(len(Y_split)))
split = X_split.loc[round(len(X_split)/2),:][1]
split1 = Y_split.loc[round(len(Y_split)/2),:][2]
cluster1 = train_data[train_data['Height']<split]
cluster2 = train_data[train_data['Height']>split]

cluster1_1 = cluster1[cluster1['Weight']<split1]
cluster2_1 = train_data[train_data['Weight']>split1]

ans2,age1 = EuclideanDistance4KNN(target_data,cluster1,3)
ans2,age2 = EuclideanDistance4KNN(target_data,cluster1,3)

# Question 2--------------------


