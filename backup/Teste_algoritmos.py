# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:57:26 2020

@author: Fernando Elias
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import Classificador_kseg_new as kseg
import time 
import matplotlib.pyplot as plt


def get_accuracy(yp, y):
    accuracy = 0
    
    
    for i in range(y.shape[0]):
        if yp[i] == y[i]:
            accuracy = accuracy + 1
    
    accuracy = accuracy/len(y)
    
    return accuracy
    
#
    
def getResults(y, yp, CP):
    from sklearn.metrics import confusion_matrix
    
    if CP == True:
        N_inl = len(np.where(y == 0)[0])
        N_out = len(y) - N_inl
        
        #inlier
        tp, fn, fp, tn = confusion_matrix(y, yp).ravel()
        prec_inl = tp/(tp + fp)
        rec_inl = tp/(tp + fn)
        f1_inl = 2*((prec_inl * rec_inl)/(prec_inl + rec_inl))    
        #outlier
        tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
        prec_out = tp/(tp + fp)
        rec_out = tp/(tp + fn)
        f1_out = 2*((prec_out * rec_out)/(prec_out + rec_out))
    
    else:
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

def calculaROC(model, CP, Xst, Xf):
    
    if (CP == True):
        
        e = model['edges']
        v = model['vertices']
    
        from copy import copy
        ii, dn = kseg.Kseg_new.map_to_arcl(copy(e),copy(v),Xst)
        ii, dp = kseg.Kseg_new.map_to_arcl(copy(e),copy(v),Xf)
        del ii 
        y_pred = np.concatenate((dn, dp), axis = 0)
        y  = np.concatenate((np.zeros(Xst.shape[0]), np.ones(Xf.shape[0])), axis = 0)
    ##
    else:
        Xt = np.concatenate((Xst, Xf), axis = 0)
        y_pred = model.score_samples(Xt)
        y  = np.concatenate((np.ones(Xst.shape[0]), np.zeros(Xf.shape[0])), axis = 0)
    ##
    
        
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(y,y_pred)
    
    return fpr, tpr, threshold

##

def geraROC(tpr_t, fpr_t, name):
    means = {}
    stds = {}
    from sklearn import metrics
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random', alpha=.8)
    
    for key in tpr_t:
        tpr = tpr_t[key]
        fpr = fpr_t[key]
        for i in range(len(tpr)):
            auc = metrics.auc(tpr[i], fpr[i])
            aucs.append(auc)
            
            interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, 
            label=r'Mean ROC %s (AUC = %0.2f $\pm$ %0.2f)' % (key, mean_auc, std_auc),
            lw=2, alpha=.8)
    
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        means.update({key: mean_auc})
        stds.update({key: std_auc})
        
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Mean ROC Curve " + name, xlabel = "False Positive Rate", ylabel = "True Positive Rate")
        
    ax.legend(loc="lower right")
    plt.show()
    return means, stds
    
##


def normalizar(x):
    n = x.shape[1]
    x_u = np.zeros((x.shape[0], x.shape[1]))
    x_n = np.zeros((x.shape[0], x.shape[1]))
    
    for i in range(n):
        x_u[:,i] = x[:,i] - np.mean(x[:,i])
        aa = np.std(x[:,i])
        if aa != 0:
            x_n[:,i] = x_u[:,i] / aa
    
    return x_n
    
#

#importando e manipulando dataset
# dataset_name = 'Credit Card Fraud Detection'
# import pandas as pd
# df = pd.read_csv('C:/Users/Fernando Elias/Documents/Datasets/310_23498_bundle_archive/creditcard.csv')
# X = df.to_numpy()


dataset_name = 'Mammography'
import scipy.io as sio 
df = sio.loadmat('mammography.mat')
df_x = df['X']
df_y = df['y']
X = np.concatenate((df_x, df_y), axis = 1)

# dataset_name = 'MNIST'
# import scipy.io as sio 
# df = sio.loadmat('mnist.mat')
# df_x = df['X']
# df_y = df['y']
# X = np.concatenate((df_x, df_y), axis = 1)

# dataset_name = 'Breast Cancer Wisconsin'
# import scipy.io as sio 
# df = sio.loadmat('breastw.mat')
# df_x = df['X']
# df_y = df['y']
# X = np.concatenate((df_x, df_y), axis = 1)

# dataset_name = 'annthyroid dataset'
# import scipy.io as sio 
# df = sio.loadmat('annthyroid.mat')
# df_x = df['X']
# df_y = df['y']
# X = np.concatenate((df_x, df_y), axis = 1)



#tamanho do conj. de treino e numero de execuções
vezes = 5
save = False


ii = np.where(X[:,-1] == 1)
ii = ii[0]

Xfi = X[np.where(X[:,-1] == 1)]
Xs = X[np.where(X[:,-1] == 0)]

Xs = np.delete(Xs, np.s_[0:1], axis = 1)
Xfi = np.delete(Xfi, np.s_[0:1], axis = 1)
Xs = np.delete(Xs, np.s_[Xs.shape[1]-1:Xs.shape[1]], axis = 1)
Xfi = np.delete(Xfi, np.s_[Xfi.shape[1]-1:Xfi.shape[1]], axis = 1)

Xf = normalizar(Xfi)

# Xfi = Xfi - np.mean(Xfi, axis = 0)
# Xf = Xfi/(np.max(np.abs(Xfi), axis = 0))

#Determinando tamanho do conjunto de treino:
Nt = int(0.7*Xs.shape[0])

#parametros de desempenho:
accSF_treino = np.zeros((vezes,3))
accSF_teste = np.zeros((vezes,3))
accFF_teste = np.zeros((vezes,3))

tprs_CP = []; fprs_CP = []; thresholds_CP = []
prec_inl_CP = []; rec_inl_CP = []; f1_inl_CP = []; acc_CP = []
prec_out_CP = []; rec_out_CP = []; f1_out_CP = []; f1_CP =  []
time_CP = []

tprs_IFO = []; fprs_IFO = []; thresholds_IFO = []
prec_inl_IFO = []; rec_inl_IFO = []; f1_inl_IFO = []; acc_IFO = [] 
prec_out_IFO = []; rec_out_IFO = []; f1_out_IFO = []; f1_IFO  = []
time_IFO = []

tprs_SVM = []; fprs_SVM = []; thresholds_SVM = []
prec_inl_SVM = []; rec_inl_SVM = []; f1_inl_SVM = []; acc_SVM = [] 
prec_out_SVM = []; rec_out_SVM = []; f1_out_SVM = []; f1_SVM  = []
time_SVM = []



for z in range(0, vezes):
    
    print('iteration number: ', z+1)
    for i in range(5):
        np.random.shuffle(Xs)
    
    Xspi = Xs[0:Nt, :]
    Xsp = normalizar(Xspi)
    
    Xsti = Xs[Nt:, :]
    Xst = normalizar(Xsti)
    
    # #CP
    ini = time.time()
    param = kseg.unsupervised_ksegFit(Xsp, 10, 1, 1, 0.15, 1000)  
    fim = time.time()
    time_CP.append(fim-ini)
    uu, accSF_treino[z, 0] = kseg.unsupervised_ksegPredict(param, Xsp)
    uu1, accSF_teste[z, 0]  = kseg.unsupervised_ksegPredict(param, Xst)
    uu2, accFF_teste[z, 0]  = kseg.unsupervised_ksegPredict(param, Xf)
    fpr, tpr, threshold = calculaROC(param, True, Xst, Xf)
    tprs_CP.append(tpr); fprs_CP.append(fpr); thresholds_CP.append(threshold)   
    
    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(
        np.concatenate((np.zeros(Xst.shape[0]), np.ones(Xf.shape[0])), axis = 0), 
        np.concatenate((uu1, uu2)), True)
    
    prec_inl_CP.append(prec_inl); rec_inl_CP.append(rec_inl); f1_inl_CP.append(f1_inl); acc_CP.append(acc)
    prec_out_CP.append(prec_out); rec_out_CP.append(rec_out); f1_out_CP.append(f1_out); f1_CP.append(f1)
    
    #Isolation Forest
    cont = 0.29
    ini = time.time()
    rng = np.random.RandomState(42)
    clf_IFO = IsolationForest(n_estimators = 100, max_samples=100,
                      random_state=rng, contamination=cont).fit(Xsp)
    
    fim = time.time()
    time_IFO.append(fim-ini)
    accSF_treino[z, 1] = get_accuracy(clf_IFO.predict(Xsp),  np.ones(Xsp.shape[0]))  
    accSF_teste[z, 1] = get_accuracy(clf_IFO.predict(Xst),   np.ones(Xst.shape[0]))
    accFF_teste[z, 1] = get_accuracy(clf_IFO.predict(Xf), -1*np.ones(Xf.shape[0]))
    fpr, tpr, threshold = calculaROC(clf_IFO, False, Xst, Xf)
    tprs_IFO.append(tpr); fprs_IFO.append(fpr); thresholds_IFO.append(threshold)  
    
    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(
        np.concatenate((np.ones(Xst.shape[0]), -1*np.ones(Xf.shape[0])), axis = 0), 
        np.concatenate((clf_IFO.predict(Xst), clf_IFO.predict(Xf))), False)
    
    prec_inl_IFO.append(prec_inl); rec_inl_IFO.append(rec_inl); f1_inl_IFO.append(f1_inl); acc_IFO.append(acc)
    prec_out_IFO.append(prec_out); rec_out_IFO.append(rec_out); f1_out_IFO.append(f1_out); f1_IFO.append(f1)
    
    # OC-SVM
    ini = time.time()
    clf_SVM = OneClassSVM(gamma='auto', nu = 0.32).fit(Xsp)
    fim = time.time()
    time_SVM.append(fim-ini)
   
    accSF_treino[z, 2] = get_accuracy(clf_SVM.predict(Xsp),  np.ones(Xsp.shape[0]))  
    accSF_teste[z, 2] = get_accuracy(clf_SVM.predict(Xst),   np.ones(Xst.shape[0]))
    accFF_teste[z, 2] = get_accuracy(clf_SVM.predict(Xf), -1*np.ones(Xf.shape[0]))
    fpr, tpr, threshold = calculaROC(clf_SVM, False, Xst, Xf)
    tprs_SVM.append(tpr); fprs_SVM.append(fpr); thresholds_SVM.append(threshold)  
    
    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(
        np.concatenate((np.ones(Xst.shape[0]), -1*np.ones(Xf.shape[0])), axis = 0), 
        np.concatenate((clf_SVM.predict(Xst), clf_SVM.predict(Xf))), False)
    
    prec_inl_SVM.append(prec_inl); rec_inl_SVM.append(rec_inl); f1_inl_SVM.append(f1_inl); acc_SVM.append(acc)
    prec_out_SVM.append(prec_out); rec_out_SVM.append(rec_out); f1_out_SVM.append(f1_out); f1_SVM.append(f1)
    
##
fprs_total = {'Principal Curves': fprs_CP, 'Isolation Forest': fprs_IFO, 'One-Class SVM': fprs_SVM}
tprs_total = {'Principal Curves': tprs_CP, 'Isolation Forest': tprs_IFO, 'One-Class SVM': tprs_SVM}
# fprs_total = {'Principal Curves': fprs_CP}
# tprs_total = {'Principal Curves': tprs_CP}

means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)
auc_CP =  [means_auc['Principal Curves'], stds_auc['Principal Curves']]
# auc_IFO = [means_auc['Isolation Forest'], stds_auc['Isolation Forest']]
# auc_SVM = [means_auc['One-Class SVM'], stds_auc['One-Class SVM']]

def exibe():
    

    print("\nPrincipal Curves:")
    print('Treino ACC:     ''%.2f' %(np.mean(accSF_treino[:,0])*100),'%.2f' %(np.std(accSF_treino[:,0])*100))
    print("----- teste ------")
    # print('Teste InLier:      ''%.2f' %(np.mean(accSF_teste[:,0])*100),'%.2f' %(np.std(accSF_teste[:,0])*100))
    # print('Teste OutLier :    ''%.2f' %(np.mean(accFF_teste[:,0])*100),'%.2f' %( np.std(accFF_teste[:,0])*100))
    # print('Precision InLier:  ''%.2f' %((np.mean(prec_inl_CP))*100),'%.2f' %((np.std(prec_inl_CP))*100))
    # print('Precision OutLier: ''%.2f' %((np.mean(prec_out_CP))*100),'%.2f' %((np.std(prec_out_CP))*100))
    print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100))
    print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100))
    print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100))
    print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100))
    print('F1-Score Model:    ''%.2f' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100))
    print('ACC Model:         ''%.2f' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100))
    print('AUC Model:         ''%.2f' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100))
    print('Time:              ''%.2f' %((np.mean(time_CP))),'%.2f' %((np.std(time_CP))))
    
    
    # print("\nIsolation Forest:")
    # print('Treino InLier:     ''%.2f' %(np.mean(accSF_treino[:,1])*100),'%.2f' %(np.std(accSF_treino[:,1])*100))
    # print('Teste InLier:      ''%.2f' %(np.mean(accSF_teste[:,1])*100),'%.2f' %(np.std(accSF_teste[:,1])*100))
    # print('Teste OutLier :    ''%.2f' %(np.mean(accFF_teste[:,1])*100),'%.2f' %( np.std(accFF_teste[:,1])*100))
    # print('Precision InLier:  ''%.2f' %((np.mean(prec_inl_IFO))*100),'%.2f' %((np.std(prec_inl_IFO))*100))
    # print('Precision OutLier: ''%.2f' %((np.mean(prec_out_IFO))*100),'%.2f' %((np.std(prec_out_IFO))*100))
    # print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_IFO))*100),'%.2f' %((np.std(rec_inl_IFO))*100))
    # print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_IFO))*100),'%.2f' %((np.std(rec_out_IFO))*100))
    # print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_IFO))*100),'%.2f' %((np.std(f1_inl_IFO))*100))
    # print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_IFO))*100),'%.2f' %((np.std(f1_out_IFO))*100))
    # print('F1-Score Model:    ''%.2f' %((np.mean(f1_IFO))*100),'%.2f' %((np.std(f1_IFO))*100))
    # print('ACC Model:         ''%.2f' %((np.mean(acc_IFO))*100),'%.2f' %((np.std(acc_IFO))*100))
    # print('AUC Model:         ''%.2f' %(auc_IFO[0]*100),'%.2f' %(auc_IFO[1]*100))
    # print('Time:              ''%.2f' %((np.mean(time_IFO))),'%.2f' %((np.std(time_IFO))))
    
       
    # print("\nOne-Class SVM:")
    # print('Treino InLier:     ''%.2f' %(np.mean(accSF_treino[:,2])*100),'%.2f' %(np.std(accSF_treino[:,2])*100))
    # print('Teste InLier:      ''%.2f' %(np.mean(accSF_teste[:,2])*100),'%.2f' %(np.std(accSF_teste[:,2])*100))
    # print('Teste OutLier :    ''%.2f' %(np.mean(accFF_teste[:,2])*100),'%.2f' %( np.std(accFF_teste[:,2])*100))
    # print('Precision InLier:  ''%.2f' %((np.mean(prec_inl_SVM))*100),'%.2f' %((np.std(prec_inl_SVM))*100))
    # print('Precision OutLier: ''%.2f' %((np.mean(prec_out_SVM))*100),'%.2f' %((np.std(prec_out_SVM))*100))
    # print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_SVM))*100),'%.2f' %((np.std(rec_inl_SVM))*100))
    # print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_SVM))*100),'%.2f' %((np.std(rec_out_SVM))*100))
    # print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_SVM))*100),'%.2f' %((np.std(f1_inl_SVM))*100))
    # print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_SVM))*100),'%.2f' %((np.std(f1_out_SVM))*100))
    # print('F1-Score Model:    ''%.2f' %((np.mean(f1_SVM))*100),'%.2f' %((np.std(f1_SVM))*100))
    # print('ACC Model:         ''%.2f' %((np.mean(acc_SVM))*100),'%.2f' %((np.std(acc_SVM))*100))
    # print('AUC Model:         ''%.2f' %(auc_SVM[0]*100),'%.2f' %(auc_SVM[1]*100))
    # print('Time:              ''%.2f' %((np.mean(time_SVM))),'%.2f' %((np.std(time_SVM))))
    
##

exibe()

if save == True:

    file_path = "C:/Users/Fernando Elias/Google Drive/Artigos_resumos/2020/Electronics Letters/testes/Resultados/"
    namefile = file_path + "resultado_" + dataset_name + ".txt"
    namefig = file_path + "ROC_" + dataset_name + ".png"
    plt.savefig(namefig)
    
    
    with open(namefile, "w") as stream:
        print('Results for dataset: ', dataset_name, file = stream)
        print("\nPrincipal Curves:", file = stream)
        print('Treino InLier.......''%.2f\t' %(np.mean(accSF_treino[:,0])*100),'%.2f' %(np.std(accSF_treino[:,0])*100), file = stream)
        print('Teste InLier........''%.2f\t' %(np.mean(accSF_teste[:,0])*100),'%.2f' %(np.std(accSF_teste[:,0])*100), file = stream)
        print('Teste OutLier.......''%.2f\t' %(np.mean(accFF_teste[:,0])*100),'%.2f' %( np.std(accFF_teste[:,0])*100), file = stream)
        print('Precision InLier....''%.2f\t' %((np.mean(prec_inl_CP))*100),'%.2f' %((np.std(prec_inl_CP))*100), file = stream)
        print('Precision OutLier...''%.2f\t' %((np.mean(prec_out_CP))*100),'%.2f' %((np.std(prec_out_CP))*100), file = stream)
        print('Recall InLier.......''%.2f\t' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100), file = stream)
        print('Recall OutLier......''%.2f\t' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100), file = stream)
        print('ACC Model...........''%.2f\t' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100), file = stream)
        print('AUC Model...........''%.2f\t' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100), file = stream)
        print('Time................''%.2f\t' %((np.mean(time_CP))),'%.2f' %((np.std(time_CP))), file = stream)
        print('--------------------------------------------------', file = stream)
        
        print("\nIsolation Forest:", file = stream)
        print('Treino InLier.......''%.2f\t' %(np.mean(accSF_treino[:,1])*100),'%.2f' %(np.std(accSF_treino[:,1])*100), file = stream)
        print('Teste InLier........''%.2f\t' %(np.mean(accSF_teste[:,1])*100),'%.2f' %(np.std(accSF_teste[:,1])*100), file = stream)
        print('Teste OutLier.......''%.2f\t' %(np.mean(accFF_teste[:,1])*100),'%.2f' %( np.std(accFF_teste[:,1])*100), file = stream)
        print('Precision InLier....''%.2f\t' %((np.mean(prec_inl_IFO))*100),'%.2f' %((np.std(prec_inl_IFO))*100), file = stream)
        print('Precision OutLier...''%.2f\t' %((np.mean(prec_out_IFO))*100),'%.2f' %((np.std(prec_out_IFO))*100), file = stream)
        print('Recall InLier:......''%.2f\t' %((np.mean(rec_inl_IFO))*100),'%.2f' %((np.std(rec_inl_IFO))*100), file = stream)
        print('Recall OutLier:.....''%.2f\t' %((np.mean(rec_out_IFO))*100),'%.2f' %((np.std(rec_out_IFO))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_IFO))*100),'%.2f' %((np.std(f1_inl_IFO))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_IFO))*100),'%.2f' %((np.std(f1_out_IFO))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_IFO))*100),'%.2f' %((np.std(f1_IFO))*100), file = stream)
        print('ACC Model:..........''%.2f\t' %((np.mean(acc_IFO))*100),'%.2f' %((np.std(acc_IFO))*100), file = stream)
        print('AUC Model:..........''%.2f\t' %(auc_IFO[0]*100),'%.2f' %(auc_IFO[1]*100), file = stream)
        print('Time................''%.2f\t' %((np.mean(time_IFO))),'%.2f' %((np.std(time_IFO))), file = stream)
        print('--------------------------------------------------', file = stream)
           
        print("\nOne-Class SVM:", file = stream)
        print('Treino InLier.......''%.2f\t' %(np.mean(accSF_treino[:,2])*100),'%.2f' %(np.std(accSF_treino[:,2])*100), file = stream)
        print('Teste InLier........''%.2f\t' %(np.mean(accSF_teste[:,2])*100),'%.2f' %(np.std(accSF_teste[:,2])*100), file = stream)
        print('Teste OutLier.......''%.2f\t' %(np.mean(accFF_teste[:,2])*100),'%.2f' %( np.std(accFF_teste[:,2])*100), file = stream)
        print('Precision InLier....''%.2f\t' %((np.mean(prec_inl_SVM))*100),'%.2f' %((np.std(prec_inl_SVM))*100), file = stream)
        print('Precision OutLier...''%.2f\t' %((np.mean(prec_out_SVM))*100),'%.2f' %((np.std(prec_out_SVM))*100), file = stream)
        print('Recall InLier:......''%.2f\t' %((np.mean(rec_inl_SVM))*100),'%.2f' %((np.std(rec_inl_SVM))*100), file = stream)
        print('Recall OutLier:.....''%.2f\t' %((np.mean(rec_out_SVM))*100),'%.2f' %((np.std(rec_out_SVM))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_SVM))*100),'%.2f' %((np.std(f1_inl_SVM))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_SVM))*100),'%.2f' %((np.std(f1_out_SVM))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_SVM))*100),'%.2f' %((np.std(f1_SVM))*100), file = stream)
        print('ACC Model...........''%.2f\t' %((np.mean(acc_SVM))*100),'%.2f' %((np.std(acc_SVM))*100), file = stream)
        print('AUC Model...........''%.2f\t' %(auc_SVM[0]*100),'%.2f' %(auc_SVM[1]*100), file = stream)
        print('Time................''%.2f\t' %((np.mean(time_SVM))),'%.2f' %((np.std(time_SVM))), file = stream)
    



