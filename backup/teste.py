# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:25:31 2024

@author: ferna
"""

import numpy as np
from Models import *
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_classes=3, n_informative=6)
y = y+1
clf = MultiClassPC(10, 1, 1, 1000)
clf.fit(X, y)
xtt = X[0:2, :]
ypp = clf.predict_proba(X)
ytt = clf.predict(X)


# from sklearn.tree import DecisionTreeClassifier
# clf2 = DecisionTreeClassifier().fit(X,y)
# clf2.predict_proba(xtt)


# def form_teste(d):
#     D = np.sum(1/d)
#     print(D)
#     inds = []
#     for di in d:
#         ind = (1/di)/D
#         print(ind)
#         inds.append(ind)
#     #
#     print(np.sum(inds))
#     return inds
# #
# d = np.array([6, 4])
# aa = form_teste(d)