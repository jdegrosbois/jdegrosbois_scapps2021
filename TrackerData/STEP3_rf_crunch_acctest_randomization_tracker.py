#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:33:10 2021

@author: john
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn import metrics

xaccuracy = []
yaccuracy = []
zaccuracy = []

for axis in ['z']:#['x','y','z']:
    print('\nStarting axis {}\n\n'.format(axis))
    for repeats in range(0,100):
        print('...Starting repeat {}'.format(repeats+1))
        all_data = pd.read_csv('{}data.csv'.format(axis))
        no_jump_all = all_data[all_data["Jump"]==0]
        yes_jump = all_data[all_data["Jump"]==1]
        temp = pd.array([True]* 166 + [False]*38)
        np.random.shuffle(temp)
        no_jump = no_jump_all[temp]

        yJ = yes_jump["Jump"]
        yD = yes_jump.iloc[:,1:]
        
        xJ = no_jump["Jump"]
        xD = no_jump.iloc[:,1:]
        
        #split the x and y data into training and test-sets
        ytrain, ytest, Jytrain, Jytest = train_test_split(yD,yJ,test_size =0.4,random_state=repeats)
        xtrain, xtest, Jxtrain, Jxtest = train_test_split(xD,xJ,test_size =0.4,random_state=repeats)
        
        #combine the test and train-data into single variables
        train_data = pd.concat([xtrain,ytrain])
        scaler_train = StandardScaler().fit(train_data)
        train_data = scaler_train.transform(train_data)
        train_labels = pd.concat([Jxtrain,Jytrain])
        test_data = pd.concat([xtest,ytest])
        scaler_test = StandardScaler().fit(test_data)
        test_data = scaler_test.transform(test_data)
        test_labels = pd.concat([Jxtest,Jytest])
        
        from sklearn.ensemble import RandomForestClassifier
        
        rows,cols = np.shape(train_data)
        accuracy = []
        depth  = [None]
        
        for d in depth:
            #clf = RandomForestClassifier(random_state=0,max_depth = d,n_estimators =2000)
            #clf.fit(train_data,train_labels)
            #print('All Data...')
            #print('Axis = {}; and Depth of {}'.format(axis,d))
            #print('Train-Score = {}'.format(clf.score(train_data,train_labels)))
            #print('Test-Score = {}\n'.format(clf.score(test_data,test_labels)))
        
            train_acc = []
            accuracy = []
            aoc= []
            for samples in range(4,cols):    
                clf = RandomForestClassifier(random_state=repeats,max_features = samples//4,max_depth = d,n_estimators =2000)
                clf.fit(train_data[:,:samples],train_labels)
                print('Axis = {}; With Sample = {}; and Depth of {}'.format(axis,samples,d))
                print('Train-Score = {}'.format(metrics.accuracy_score(train_labels,clf.predict(train_data[:,:samples]))))
                print('Test-Score = {}'.format(metrics.accuracy_score(test_labels,clf.predict(test_data[:,:samples]))))
                #print('AOC = {}\n'.format(roc_auc_score(test_labels,clf.predict(test_data[:,:samples]))))
                accuracy.append(clf.score(test_data[:,:samples],test_labels))
                train_acc.append(clf.score(train_data[:,:samples],train_labels))
                aoc.append(roc_auc_score(test_labels,clf.predict(test_data[:,:samples])))
                #plt.plot(clf.feature_importances_)
                #plt.plot([0,38],[clf.feature_importances_[0],clf.feature_importances_[0]],color='red')
                #plt.title('Random Forest Importances')
                #plt.show()
            
            xticks = [-2+i*8 for i in range(1,len(accuracy)+1)]
            #plt.plot(xticks,accuracy,color='red',alpha=0.5)
            #plt.plot(xticks,accuracy,'*')
            #plt.plot(xticks,train_acc,color='blue',alpha=0.5)
            #plt.plot(xticks,train_acc,'.')
            #plt.xlabel('Included Time (ms')
            #plt.ylabel('Prediction AUC')
            #plt.title('Axis = {}; Max-leaf-depth = {}'.format(axis,d))
            #plt.show()
            #plt.plot(xticks,aoc)
            #plt.title('AUC Across Time: Axis = {}'.format(axis))
            #plt.show()
        if axis == 'x':
            xaccuracy.append(accuracy)
        elif axis == 'y':
            yaccuracy.append(accuracy)
        elif axis == 'z':
            zaccuracy.append(accuracy)
    if axis == 'x':
       xdf = pd.DataFrame(xaccuracy)
       xdf.to_csv('x_accuracy_bootstrap.csv',index=False,header=False)
    elif axis == 'y':
       ydf = pd.DataFrame(yaccuracy)
       ydf.to_csv('y_accuracy_bootstrap.csv',index=False,header=False)
    elif axis == 'z':
       rdf = pd.DataFrame(zaccuracy)
       rdf.to_csv('z_accuracy_bootstrap.csv',index=False,header=False)

tdf = pd.DataFrame(xticks)
tdf.to_csv('time_ticks.csv',index=False,header=False)
