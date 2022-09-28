#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:00:14 2022

@author: felias
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:26:16 2021

@author: Fernando Elias
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import Classificador_kseg_new as kseg
import time 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from copy import copy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import pickle

def getResults(y, yp):
    
    
    
    N_inl = len(np.where(y == 1)[0])
    N_out = len(y) - N_inl
    
    #inlier
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    prec_inl = tp/(tp + fp)
    rec_inl = tp/(tp + fn)
    f1_inl = 2*((prec_inl * rec_inl)/(prec_inl + rec_inl))
    #outlier
    tp, fn, fp, tn = confusion_matrix(y, yp).ravel()
    prec_out = tp/(tp + fp)
    rec_out = tp/(tp + fn)
    f1_out = 2*((prec_out * rec_out)/(prec_out + rec_out))    
    
    f1 = (f1_inl*N_inl + f1_out*N_out)/(N_inl + N_out)
    acc = (tp+tn)/(tp+tn+fp+fn)
    return prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc
##


def calculaROC(model, CP, X):
    
    if (CP == True):
        
        e = model['edges']
        v = model['vertices']
    
        ii, y_pred = kseg.Kseg_new.map_to_arcl(copy(e),copy(v), X[:,0:-1])
        del ii 
        y  = X[:,-1]
        tpr, fpr, threshold = metrics.roc_curve(y, y_pred)
    ##
    else:
        y_pred = model.score_samples(X[:,0:-1])
        y  = X[:,-1]
        fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
    ##
         
    return fpr, tpr, threshold

##

def geraROC(tpr_t, fpr_t, name):
    means = {}
    stds = {}
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random', alpha=.8)
    
    for key in tpr_t:
        tprs = []
        aucs = []
        
        tpr = tpr_t[key]
        fpr = fpr_t[key]
        for i in range(len(tpr)):
            auc = metrics.auc(fpr[i], tpr[i])
            aucs.append(auc)
            
            interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        means.update({key: mean_auc})
        stds.update({key: std_auc})
        
        ax.plot(mean_fpr, mean_tpr, 
            label=r'Mean ROC %s' % (key),
            lw=2, alpha=.8)
    
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        
        del mean_auc, std_auc
    ##
        
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Mean ROC Curve " + name, xlabel = "False Positive Rate", ylabel = "True Positive Rate")
        
    ax.legend(loc="lower right")
    plt.show()
    return means, stds
    
##

data_path  = "/home/felias/Documentos/Letters_testes/data/"
path = data_path + 'teste_Flame Norm.mat'
data = sio.loadmat(path)

dataset_name = 'synthetic_data'

teste = data['dados_teste']
c0_teste  = teste[(np.where(teste[:,-1]==1)[0]), :]
c1_teste  = teste[(np.where(teste[:,-1]==-1)[0]), :]
Xsp = data['dados_treino']
X_teste = np.concatenate((c0_teste, c1_teste), axis = 0)

Nt = Xsp.shape[0]

# model_path = "/home/felias/Documentos/Letters_testes/modelos/"

# # dataset_name = 'annthyroid dataset'
# # dataset_name = 'Breast Cancer Wisconsin'
# # dataset_name = 'MNIST'
# dataset_name = 'Mammography'
# # dataset_name = 'Flame Norm'

# path = model_path+ 'CP_' + dataset_name+ '.mat'
# param = sio.loadmat(path)
# curve = {'edges': param['edges'], 'vertices': param['vertices']}

# data_path = "/home/felias/Documentos/Letters_testes/data/"
# path  = data_path + 'teste_'+ dataset_name +'.mat'

# data = sio.loadmat(path)

# X_teste = data['dados_teste']
# Xsp    = data['dados_treino']

is_curve = False

#%%
#tamanho do conj. de treino e numero de execuções
vezes = 10
save = True
CP  = 1


#CP params:
k_max = 10; 

k_ = np.arange(1, 21, 1)
ol = 0.1

for i in range(5):
    np.random.shuffle(X_teste)
#


#parametros de desempenho:

tprs_ = []; fprs_= []; thresholds_ = []
prec_inl_ = []; rec_inl_ = []; f1_inl_ = []; acc_ = []
prec_out_ = []; rec_out_ = []; f1_out_ = []; f1_ =  []
time_ = []


    
#%%
#parametros de desempenho:
acc_treino_ = []

#modelos:
models_ = []

kf = KFold(n_splits = vezes, shuffle=True)


fold = 0
for train_index, test_index in kf.split(Xsp):
    print('fold = ', fold); fold = fold+1
    X_train, X_test = Xsp[train_index], Xsp[test_index]
    acc_treino_CP  = []; models_CP = []
    
    for i in range(len(k_)):
        param = kseg.unsupervised_ksegFit(X_train, k_[i], 1, 1, ol, 1000, False, 0)
        models_CP.append(copy(param))
        saida_val = kseg.unsupervised_ksegPredict(param, X_test)
        acc = np.mean(saida_val == np.ones(saida_val.shape))
        acc_treino_CP.append(acc)
        del param
        ##
    ##
    models_.append(models_CP)
    acc_treino_.append(acc_treino_CP)
    del acc_treino_CP, models_CP
##
##


    


models_u = copy(models_); del models_
acc_treino_u = copy(acc_treino_); del acc_treino_

models_ = []; acc_treino_ = []

for i in range(len(k_)):
    model_ = []
    acc_t   = []
    for j in range(len(models_u)):
        model_.append(models_u[j][i])
        acc_t.append(acc_treino_u[j][i])
    ##
    models_.append(model_)
    acc_treino_.append(acc_t)
    del model_, acc_t
##



#%%
bests_ = []


for i in range(len(k_)):
    acc_treino_CP = acc_treino_[i]
    models_CP = models_[i]
    best_cp = np.argmax(acc_treino_CP); param = models_CP[best_cp]
    bests_.append(param)
    del param, acc_treino_CP, models_CP
##

for i in range(len(k_)):   
    tprs_CP = []; fprs_CP = []; thresholds_CP = []
    param = bests_[i]
    ini = time.time()
    saida_teste = kseg.unsupervised_ksegPredict(param, X_teste[:,0:-1])
    fim = time.time()
    time_.append(fim-ini)
    
    fpr, tpr, threshold = calculaROC(param, True, X_teste)
    tprs_CP.append(tpr); fprs_CP.append(fpr); thresholds_CP.append(threshold)
    tprs_.append(tprs_CP); fprs_.append(fprs_CP); thresholds_.append(thresholds_CP)   
    del tprs_CP; fprs_CP; thresholds_CP
    
    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste) 
    prec_inl_.append(prec_inl); rec_inl_.append(rec_inl); f1_inl_.append(f1_inl); acc_.append(acc)
    prec_out_.append(prec_out); rec_out_.append(rec_out); f1_out_.append(f1_out); f1_.append(f1)
    ##
    
##  



#%%
auc_ = []
fprs_total = {}; tprs_total = {}

for i in range(len(k_)):
    fprs_CP = fprs_[i]; tprs_CP = tprs_[i]; thresholds_CP = thresholds_[i]
    key = 'One-Class PC: k_ = ' + str(k_[i])
    fprs_total.update({key: fprs_CP})
    tprs_total.update({key: tprs_CP})
##

means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)

for i in range(len(k_)):
    key = 'One-Class PC: k_ = ' + str(k_[i])
    auc_CP =  [means_auc[key], stds_auc[key]]
    auc_.append(auc_CP)
##

 

def exibe(acc_treino_CP, rec_inl_CP, rec_out_CP, f1_inl_CP, f1_out_CP,
          f1_CP, acc_CP, auc_CP, time_CP, k_):
    
    print("\nOne-Class One-Class PC:", "k_ = ", k_)
    print('Treino ACC:     ''%.2f' %((np.mean(acc_treino_CP))*100),'%.2f' %((np.std(acc_treino_CP))*100))
    print("----- teste ------")
    print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100))
    print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100))
    print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100))
    print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100))
    print('F1-Score Model:    ''%.2f' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100))
    print('ACC Model:         ''%.2f' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100))
    print('AUC Model:         ''%.2f' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100))
    print('Time:              ''%.2f' %((np.mean(time_CP))),'%.2f' %((np.std(time_CP))))

##

print('dataset: ', dataset_name)

for i in range(len(k_)):
    acc_treino_CP = acc_treino_[i]; rec_inl_CP = rec_inl_[i]; rec_out_CP = rec_out_[i]
    f1_inl_CP = f1_inl_[i]; f1_out_CP  = f1_out_[i]
    f1_CP = f1_[i]; acc_CP = acc_[i]; auc_CP = auc_[i]; time_CP = time_[i]
    
    exibe(acc_treino_CP, rec_inl_CP, rec_out_CP, f1_inl_CP, f1_out_CP,
          f1_CP, acc_CP, auc_CP, time_CP, k_[i])

if(save):
    arq = '/home/felias/Documentos/Letters_testes/modelos/'
    arq = arq+ dataset_name + '_k_ol_01.pkl'
    pickle.dump(bests_, open(arq, 'wb'))
##

#%%
fig, ax = plt.subplots(dpi = 150)
ax.plot(k_, (rec_inl_), 'b.-', label = 'Known Class')
ax.plot(k_, (rec_out_), 'r.-', label = 'Outliers')
plt.legend(loc='lower right', fontsize = 10)
plt.xlabel('Number of segments (k)', fontsize = 14)
plt.ylabel('Recall', fontsize = 14)
plt.xticks(np.arange(1, 23, 2))
plt.grid(color='gray', linestyle='-', linewidth=0.3, alpha = 0.3)
#%%
'''
if save == True:
    import pickle
    
    file_path = "C:/Users/Fernando Elias/Google Drive/Artigos_resumos/2020/Electronics Letters/testes/Resultados/"
    namefile = file_path + "resultado_teste_k__" + dataset_name + ".txt"
    namefig = file_path + "ROC_" + dataset_name + ".png"
    plt.savefig(namefig)
    
    name_model_CP  = "CP_"  + dataset_name + ".mat"
    
    
    sio.savemat(name_model_CP, param)  
  
    
    nome = 'teste_'+dataset_name+'.mat'
    sio.savemat(nome, {'dados_treino': Xsp, 'dados_teste': X_teste})
    
    with open(namefile, "w") as stream:
        print('Results for dataset: ', dataset_name, file = stream)
        print("\nOne-Class PC:", file = stream)
        print('Treino ACC:     ''%.2f' %(np.max(acc_treino_CP)*100), file = stream)
        print("----- teste ------", file = stream)
        print('Recall InLier.......''%.2f\t' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100), file = stream)
        print('Recall OutLier......''%.2f\t' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100), file = stream)
        print('ACC Model...........''%.2f\t' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100), file = stream)
        print('AUC Model...........''%.2f\t' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100), file = stream)
        print('Time................''%.2f\t' %((np.mean(time_CP))),'%.2f' %((np.std(time_CP))), file = stream)
        print('--------------------------------------------------', file = stream)
        
##
'''