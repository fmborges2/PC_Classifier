#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:45:38 2022

@author: felias
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

model_path = "/home/felias/Documentos/Letters_testes/modelos/"

#Pricipal Curves
path = model_path+'CP_Flame Norm.mat'
param = sio.loadmat(path)

e = param['edges']
v = param['vertices']


ws = 5; Cs = 'k'
wi = 2; Ci = 'k'

for i in range(1, e.shape[0]):
    print(i)
    j = np.where(e[:,i] == 2)[0]
    if j.shape[0] != 0:
        plt.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs)
    ##
    j = np.where(e[:,i] == 1)[0]
    if j.shape[0] != 0:
        plt.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = wi, color = Ci)
    ##
##





