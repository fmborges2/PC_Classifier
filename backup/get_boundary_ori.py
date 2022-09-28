# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:38:02 2021

@author: Fernando Elias
"""

import pickle 
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import Classificador_kseg_new as kseg

data_path  = "/home/felias/Documentos/Letters_testes/data/"
path = data_path + 'teste_Original Flame Data.mat'
data = sio.loadmat(path)


teste = data['dados_teste']
c0_teste  = teste[(np.where(teste[:,-1]==1)[0]), :]
c1_teste  = teste[(np.where(teste[:,-1]==-1)[0]), :]
c0_treino = data['dados_treino']


model_path = "/home/felias/Documentos/Letters_testes/modelos/"

#Pricipal Curves
path = model_path+'CP_Original Flame Data.mat'
param = sio.loadmat(path)

fig, ax = plt.subplots(dpi = 150)
ax.plot(c0_treino[:,0], c0_treino[:,1], 'bo', alpha = 0.3, label = 'training data (known class)')
ax.plot(c0_teste[:,0], c0_teste[:,1], 'kv', alpha = 0.3, label = 'test data (known class)')
ax.plot(c1_teste[:,0], c1_teste[:,1], 'r*', alpha = 0.3, label = 'outliers')
X = c0_treino

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
cs = ax.contour(xx, yy, Z, levels=[0], cmap = 'Purples_r')
cs.collections[0].set_label('boundary: One-Class PC')
plt.legend(loc='lower right', fontsize = 6)
plt.xlabel('$x_1$', fontsize = 12)	
plt.ylabel('$x_2$', fontsize = 12)
plt.xlim([-2.15, 2.15])
plt.ylim([-2.5 , 4.15])
plt.show()


#Isolation Forest
path = model_path+'IFO_Original Flame Data.pkl'
clf = pickle.load(open(path,'rb'))

fig, ax = plt.subplots(dpi = 150)
ax.plot(c0_treino[:,0], c0_treino[:,1], 'bo', alpha = 0.3, label = 'training data (known class)')
ax.plot(c0_teste[:,0], c0_teste[:,1], 'kv', alpha = 0.3, label = 'test data (known class)')
ax.plot(c1_teste[:,0], c1_teste[:,1], 'r*', alpha = 0.3, label = 'outliers')
X = c0_treino

h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs = ax.contour(xx, yy, Z, levels=[0], cmap = 'Purples_r')
cs.collections[0].set_label('boundary: Isolation Forest')
plt.legend(loc='lower right', fontsize = 6)
plt.xlabel('$x_1$', fontsize = 12)	
plt.ylabel('$x_2$', fontsize = 12)
plt.xlim([-2.15, 2.15])
plt.ylim([-2.5 , 4.15])
plt.show()


#OC - SVM
path = model_path+'SVM_Original Flame Data.pkl'
clf = pickle.load(open(path,'rb'))

fig, ax = plt.subplots(dpi = 150)
ax.plot(c0_treino[:,0], c0_treino[:,1], 'bo', alpha = 0.3, label = 'training data (known class)')
ax.plot(c0_teste[:,0], c0_teste[:,1], 'kv', alpha = 0.3, label = 'test data (known class)')
ax.plot(c1_teste[:,0], c1_teste[:,1], 'r*', alpha = 0.3, label = 'outliers')
X = c0_treino

h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs = ax.contour(xx, yy, Z, levels=[0], cmap = 'Purples_r')
cs.collections[0].set_label('boundary: One-Class SVM')
plt.legend(loc='lower right', fontsize = 6)
plt.xlabel('$x_1$', fontsize = 12)	
plt.ylabel('$x_2$', fontsize = 12)
plt.xlim([-2.15, 2.15])
plt.ylim([-2.5 , 4.15])
plt.show()