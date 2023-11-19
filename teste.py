# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:09:08 2023

@author: ferna
"""

from classifier import *

import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'Flame Norm'
import scipy.io as sio 
df = sio.loadmat('flame_data.mat')
df_x = df['X']
df_y = df['y']


X = np.concatenate((df_x, df_y), axis = 1)

x = X[0:1000, 0:2]
cp = Kseg(10, 1, 1, 1000)

cp.fitCurve(x)

fig, ax = plt.subplots(dpi = 150)
plt.plot(x[:,0], x[:,1], 'o')
cp.plot_curve(ax)
ax.legend()


y = X[:,2]
X = X[:,0:2]

c0_x = X[y == 1]
c1_x = X[y == -1]

c0_y = y[y==1]
c0_y = y[y==-1]


oc_clf = OneClassPC(10, 1, 1, 0.05, 1000)
oc_clf.fit(c0_x)
y0_p = oc_clf.predict(c0_x)
y1_p = oc_clf.predict(c1_x)


## 
y = np.zeros((X.shape[0],2))
y[0:1001, 0] = 1
y[1001:, 1]  = 1
mc_clf = MultiClassPC(10, 1, 1, 1000)
mc_clf.fit(X, y)
yp = mc_clf.predict(X)

aa = mc_clf.curves
c1 = aa[0]
c2 = aa[1]


fig,ax = plt.subplots()
c1.plot_curve(ax)

fig,ax = plt.subplots()
c2.plot_curve(ax)

print(mc_clf.score(y, yp))

