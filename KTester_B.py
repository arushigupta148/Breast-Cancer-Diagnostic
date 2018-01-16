#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:53:17 2016

@author: arushigupta148
"""

import pandas as pd
import math

df = pd.read_csv("Compressed_Data.csv")
df = df.drop(df.columns[0], axis=1)

df_m = pd.read_csv("Benign.csv")
df_m= df_m.drop(labels = ['id','diagnosis'], axis=1)

from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca.fit(df_m)
PCA(copy=True, n_components=2, whiten=False)
T=pca.transform(df_m)
df_m=pd.DataFrame(T)
df_m.columns=['col1','col2']

dist_list = []

for i in range(len(df_m)):
    new_list = []
    for j in range(len(df)):
        x = math.sqrt((float(df.ix[j][0]) - float(df_m.ix[i][0]))**2 + (float(df.ix[j][1]) - float(df_m.ix[i][1]))**2)
        new_list.append(x)
    dist_list.append(new_list)

k_list = []
mint = 2    
maxt = 30

for l in range(mint,maxt):
    correct = 0
    for i in range(len(df_m)):
        list1 = list(dist_list[i])
        list1.sort()
        list1 = list(list1[0:l])
        list2 = []
        for j in range(len(list1)):
            y = dist_list[i].index(list1[j])
            c = df.ix[y][2]
            list2.append(c)
        M_count = 0
        B_count = 0
        for k in range(len(list2)):
            if list2[k] == 'B':
                M_count += 1
            else:
                B_count += 1
        if M_count > B_count:
            correct += 1
    
    print "Accuracy for " + str(l) + " neighbors is - " + str(100*correct/15.0) + " percent"
    k_list.append(correct/15.0)

maxi = max(k_list)
ind = k_list.index(maxi)
print "\nMax accuracy with " + str(ind+mint) + " neighbors"
print "Accuracy rate obtained - " + str(maxi * 100) + " percent"