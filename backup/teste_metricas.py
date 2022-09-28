# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:59:59 2020

@author: Fernando Elias
"""




import numpy as np
from sklearn.ensemble import IsolationForest
import scipy.io as sio 
Xsp = sio.loadmat('Xs_p.mat')
Xsp = Xsp['Xs_p']
Xst = sio.loadmat('Xs_t.mat')
Xst = Xst['Xs_t'] 
Xff = sio.loadmat('Xf_t.mat')
Xff = Xff['Xf_t']


clf = IsolationForest( max_samples=100,
                      random_state=0, contamination='auto')
clf.fit(Xsp)

Xt = np.concatenate((Xst, Xff), axis = 0)
y  = np.concatenate((np.zeros(Xst.shape[0]), np.ones(Xff.shape[0])), axis = 0)
y_pred = clf.score_samples(Xt)



from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y,y_pred)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'k-', lw=2)
plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], 'b--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()





import numpy as np
import scipy.io as sio
import Classificador_kseg_new as kseg
from copy import copy 

x = sio.loadmat('dados_tst.mat')
Xss = x['Xss']
Xff = x['Xff']
del x 
np.random.shuffle(Xss)
Nt = 200
Xsp = Xss[0:Nt,:]
Xst = Xss[Nt:,:]

e, v = kseg.Kseg_new.fitCurve(Xsp, 10, 1, 1, 1000)
ii, dn = kseg.Kseg_new.map_to_arcl(copy(e),copy(v),Xst)
ii, dp = kseg.Kseg_new.map_to_arcl(copy(e),copy(v),Xff)
del ii 
y_pred = np.concatenate((dn, dp), axis = 0)

y  = np.concatenate((np.zeros(Xst.shape[0]), np.ones(Xff.shape[0])), axis = 0)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y,y_pred)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'k-', lw=2)
plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], 'b--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
print(metrics.auc(fpr, tpr))
