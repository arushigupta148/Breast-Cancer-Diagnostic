#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:21:43 2016

@author: arushigupta148
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("data.csv",header = 0)

df= df.drop(labels = ['id'], axis=1)

df1=df[df.diagnosis == 'M']
df2=df[df.diagnosis == 'B']

df1= df1.drop(labels = ['diagnosis'], axis=1)
df2= df2.drop(labels = ['diagnosis'], axis=1)

from sklearn.decomposition import PCA
pca= PCA(n_components=3)
pca.fit(df1)
PCA(copy=True, n_components=3, whiten=False)
T=pca.transform(df1)
df1=pd.DataFrame(T)
df1.columns=['col1','col2','col3']

pca= PCA(n_components=3)
pca.fit(df2)
PCA(copy=True, n_components=3, whiten=False)
T=pca.transform(df2)
df2=pd.DataFrame(T)
df2.columns=['col1','col2','col3']

fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df1.col1,df1.col2,df1.col3, c='r', label='malignant')
ax.scatter(df2.col1,df2.col2,df2.col3, c='b', label='benign')
plt.show()
"""

fig, ax = plt.subplots()
ax.scatter(df1.col1, df1.col2, c='r', label='malignant')
ax.scatter(df2.col1, df2.col2, c='b', label='benign')
plt.show()
"""
print len(df1)
print len(df2)

df = pd.concat([df1,df2])

print len(df)

class_label = []
for i in range(202):
    class_label.append('M')
for i in range(342):
    class_label.append('B')

df['diagnosis'] = class_label

df.to_csv("Compressed_Data.csv")