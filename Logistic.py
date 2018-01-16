# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import math

df = pd.read_csv("Compressed_Data.csv")
df = df.drop(df.columns[0], axis=1)

df.col1 = (df.col1 - df.col1.mean())/df.col1.std(ddof=0)
df.col2 = (df.col2 - df.col2.mean())/df.col2.std(ddof=0)

m = len(df)
alpha=0.5
theta = [0.059934091281950624, 0.66013667247412766, 2.2240077122167441, 5.200267436045749, -0.34186706640479331, 0.72338346402593545, -0.51447600601326748, 1.1963604736003937, 0.9451943759502669, -3.0331260569314291, 0.11419621609911793, 0.061088064230656512, 0.42762709404695665, 0.12808445207127342, 0.089804995954129385]
cost = 1720.0
prev_cost = 1721.0
new_list = []
for i in range(len(df)):
    new_list.append(0.0)

df1 = pd.DataFrame()
df1['col1'] = new_list
df1['diagnosis'] = df['diagnosis']    

while cost > 40.0 and cost < prev_cost:
    print "Theta vector is - " + str(theta)
    for i in range(len(df)):
        z = theta[0] * df.ix[i][0] + theta[1] * df.ix[i][1] + theta[2] * df.ix[i][0] ** 2 + theta[3] * df.ix[i][1] ** 2 + theta[4] * df.ix[i][0] * df.ix[i][1]
        z += theta[5] * df.ix[i][0] ** 3 + theta[6] * df.ix[i][0] ** 2 * df.ix[i][1] + theta[7] * df.ix[i][0] * df.ix[i][1] ** 2 + theta[8] * df.ix[i][1] ** 3 + theta[9]
        z += theta[10] * df.ix[i][0] ** 4 + theta[11] * df.ix[i][0] ** 3 * df.ix[i][1] + theta[12] * df.ix[i][0] ** 2 * df.ix[i][1] ** 2 + theta[13] * df.ix[i][0] * df.ix[i][1] ** 3 + theta[14] * df.ix[i][1] ** 4
        
        hx = 1/(1+math.exp(-z))
        df1.set_value(i,'col1',hx)
    
    prev_cost = cost
    cost = 0

    for j in range(len(df1)):
        if df1.ix[j][1] == 'M':
            y = 1
        else:
            y=0
        
        cost += (-y*math.log(df1.ix[j][0],10) - ((1-y)*math.log((1.00001-df1.ix[j][0]),10)))
    print "Cost is : " + str(cost)       
    J = -(1/float(m))*cost
    print "J value is : " + str(J)
    grad = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    for i in range(len(df1)):
        if df1.ix[i][1] == 'M':
            y = 1
        else:
            y=0
        grad[9] += df1.ix[i][0] - y
        grad[0] += (df1.ix[i][0] - y) * df.ix[i][0]
        grad[1] += (df1.ix[i][0] - y) * df.ix[i][1]
        grad[2] += (df1.ix[i][0] - y) * df.ix[i][0] ** 2
        grad[3] += (df1.ix[i][0] - y) * df.ix[i][1] ** 2
        grad[4] += (df1.ix[i][0] - y) * df.ix[i][0] * df.ix[i][1]
        grad[5] += (df1.ix[i][0] - y) * df.ix[i][0] ** 3
        grad[6] += (df1.ix[i][0] - y) * df.ix[i][0] ** 2 * df.ix[i][1]
        grad[7] += (df1.ix[i][0] - y) * df.ix[i][0] * df.ix[i][1] ** 2
        grad[8] += (df1.ix[i][0] - y) * df.ix[i][1] ** 3
        grad[10] += (df1.ix[i][0] - y) * df.ix[i][0] ** 4
        grad[11] += (df1.ix[i][0] - y) * df.ix[i][0] ** 3 * df.ix[i][1]
        grad[12] += (df1.ix[i][0] - y) * df.ix[i][0] ** 2 * df.ix[i][1] ** 2
        grad[13] += (df1.ix[i][0] - y) * df.ix[i][0] * df.ix[i][1] ** 3
        grad[14] += (df1.ix[i][0] - y) * df.ix[i][1] ** 4
    
    for i in range(len(theta)):
        theta[i] = theta[i] - (alpha * grad[i] * (1/float(m)))
print theta