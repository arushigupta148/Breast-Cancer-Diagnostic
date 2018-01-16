#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:41:10 2016

@author: akashsrihari
"""
import pandas as pd
import math

df1 = pd.read_csv("Benign.csv")
df1 = df1.drop(labels = ['id','diagnosis'], axis=1)
df2 = pd.read_csv("Malignant.csv")
df2 = df2.drop(labels = ['id','diagnosis'], axis=1)

df = pd.concat([df1,df2])

from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca.fit(df)
T=pca.transform(df)
df=pd.DataFrame(T)
df.columns=['col1','col2']

df.col1 = (df.col1 - df.col1.mean())/df.col1.std(ddof=0)
df.col2 = (df.col2 - df.col2.mean())/df.col2.std(ddof=0)

df = df[0:15]

theta = [0.059934091281950624, 0.66013667247412766, 2.2240077122167441, 5.200267436045749, -0.34186706640479331, 0.72338346402593545, -0.51447600601326748, 1.1963604736003937, 0.9451943759502669, -3.0331260569314291, 0.11419621609911793, 0.061088064230656512, 0.42762709404695665, 0.12808445207127342, 0.089804995954129385]

correct = 0

for i in range(len(df)):
    hx = theta[0] * df.ix[i][0] + theta[1] * df.ix[i][1] + theta[2] * df.ix[i][0] ** 2 + theta[3] * df.ix[i][1] ** 2 + theta[4] * df.ix[i][0] * df.ix[i][1]
    hx += theta[5] * df.ix[i][0] ** 3 + theta[6] * df.ix[i][0] ** 2 * df.ix[i][1] + theta[7] * df.ix[i][0] * df.ix[i][1] ** 2 + theta[8] * df.ix[i][1] ** 3 + theta[9]
    hx += theta[10] * df.ix[i][0] ** 4 + theta[11] * df.ix[i][0] ** 3 * df.ix[i][1] + theta[12] * df.ix[i][0] ** 2 * df.ix[i][1] ** 2 + theta[13] * df.ix[i][0] * df.ix[i][1] ** 3 + theta[14] * df.ix[i][1] ** 4
    hx = 1/(1+math.exp(-hx))
    print hx
    if hx <= 0.5:
        correct += 1
        
print "Accuracy is : " + str(correct * 100.0/15.0) + " percent"