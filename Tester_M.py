#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:12:58 2016

@author: arushigupta148
"""

import pandas as pd
import math

df = pd.read_csv("Compressed_Data.csv")
df = df.drop(df.columns[0], axis=1)

df_m = pd.read_csv("Malignant.csv")
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

correct = 0    

for i in range(len(df_m)):
    x = min(dist_list[i])
    y = dist_list[i].index(x)
    if df.ix[y][2] == 'M':
        correct += 1

print "accuracy is " + str(correct/10.0)
        