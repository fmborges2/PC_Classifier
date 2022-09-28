#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:59:34 2022

@author: felias
"""

import numpy as np
import matplotlib.pyplot as plt 
import Classificador_kseg_new as kseg

x = np.random.randn(200,1)
y = np.random.randn(200,1)
X = np.concatenate((x,y), axis =1)



def plot_boundary(X, param, color):
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = kseg.unsupervised_ksegPredict(param, np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = ax.contour(xx, yy, Z, levels=[0], cmap = color, linewidths = 0.5, linestyles = "solid")
    cs.collections[0].set_label('boundary: ol = %.2f' %(param['outlier_rate']))

##

def plot_curve(e,v):
    ws = 5; Cs = 'k'
    wi = 2; Ci = 'k'

    key_s = True
    key_i = True


    for i in range(1, e.shape[0]):
        j = np.where(e[:,i] == 2)[0]
        if j.shape[0] != 0:
            if(key_s):
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs, label = 'PC segment')
                key_s = False
            else:
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs)
        ##
        j = np.where(e[:,i] == 1)[0]
        if j.shape[0] != 0:
            if(key_i):
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = wi, color = Ci, label = 'segment connection')
                key_i = False
            else: 
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = wi, color = Ci)
            ##
        ##
    ##
##

fig, ax = plt.subplots(dpi = 150)
ax.plot(x,y, 'bo')


# ol = np.arange(0, 0.3, 0.05)
ol = np.array([0, 0.01, 0.05, 0.1, 0.15, 0.2])

param = kseg.unsupervised_ksegFit(X, 5, 1, 1, ol[0], 1000, False, 0)

e = param['edges']; v = param['vertices']

from copy import copy
e1 = copy(e); v1 = copy(v)

pos = np.where(np.sum(e1, axis = 0) == 2)[0]
e1[pos[0], pos[1]] = 1
e1[pos[1], pos[0]] = 1

# curve = {'edges': param['edges'], 'vertices': param['vertices']}
curve = {'edges': e1, 'vertices': v1}



colors = ['Purples_r', 'Greys_r', 'Greens_r', 'Reds_r', 'cool_r', 'spring_r']

plot_boundary(X, param, colors[0])

params = []
params.append(param)

for i in range(1, ol.shape[0]):
    param_ = kseg.unsupervised_ksegFit(X, 0, 0, 0, ol[i], 1000, True, curve)
    plot_boundary(X, param_, colors[i])
    params.append(param_)
    del param_
##

plot_curve(e1, v1)

# plt.legend(loc='lower right', fontsize = 6)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='upper left', fontsize = 6, bbox_to_anchor=(1, 0.5))
plt.xlabel('$x_1$', fontsize = 12)	
plt.ylabel('$x_2$', fontsize = 12)
# plt.xlim([-2.55, 2.55])
# plt.ylim([-2.2 , 3.1])
plt.show()

'''
from copy import copy
e1 = copy(e); v1 = copy(v)

pos = np.where(np.sum(e1, axis = 0) == 2)[0]
e1[pos[0], pos[1]] = 1
e1[pos[1], pos[0]] = 1

fig, ax = plt.subplots(dpi = 150)
ax.plot(x,y, 'bo')
plot_curve(e1, v1)
'''