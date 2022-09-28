#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:10:40 2022

@author: felias
"""



import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
import Classificador_kseg_new as kseg
import pickle


def plot_boundary(X, param, color, w):
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
    cs = ax.contour(xx, yy, Z, levels=[0], cmap = color, linewidths = w, linestyles = "solid")
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


data_path  = "C:/Users/Fernando Elias/Documents/GitHub/PC_Classifier/data/"
path = data_path + 'teste_Synthetic_Data.mat'
data = sio.loadmat(path)


teste = data['dados_teste']
c0_teste  = teste[(np.where(teste[:,-1]==1)[0]), :]
c1_teste  = teste[(np.where(teste[:,-1]==-1)[0]), :]
c0_treino = data['dados_treino']

model_path = "C:/Users/Fernando Elias/Documents/GitHub/PC_Classifier/models/"

#Pricipal Curves
path = model_path+'synthetic_data_ol_k_9.pkl'
models = pickle.load(open(path,'rb'))

#%%
e = models[0]['edges']; v = models[0]['vertices']

fig, ax = plt.subplots(dpi = 150)


#plot dados:
ax.plot(c0_treino[:,0], c0_treino[:,1], 'bo', alpha = 0.3, label = 'training data (known class)')
ax.plot(c0_teste[:,0], c0_teste[:,1], 'kv', alpha = 0.3, label = 'test data (known class)')
ax.plot(c1_teste[:,0], c1_teste[:,1], 'r*', alpha = 0.3, label = 'outliers')
X = c0_treino

plot_curve(e, v)

colors = ['Spectral', 'Purples_r', 'Greys_r', 'Greens_r', 'Reds_r', 'cool_r']
widths = [1, 1, 1, 2, 2, 2]

for i in range(len(models)):
    param_ = models[i]
    plot_boundary(X, param_, colors[i], widths[i])
    del param_


plt.legend(loc='lower right', fontsize = 6)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='upper left', fontsize = 6, bbox_to_anchor=(1, 0.5))
plt.xlabel('$x_1$', fontsize = 12)	
plt.ylabel('$x_2$', fontsize = 12)
plt.xlim([-2.55, 4.05])
plt.ylim([-2.2 , 3.1])
plt.show()

