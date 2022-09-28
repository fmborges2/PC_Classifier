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
            label=r'ROC %s (AUC = %0.2f)' % (key, mean_auc),
            lw=2, alpha=.8)
    
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        
        del mean_auc, std_auc
    ##
        
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="ROC Curve " + name, xlabel = "False Positive Rate", ylabel = "True Positive Rate")
        
    ax.legend(loc="lower right")
    plt.show()
    return means, stds
    
##


dataset_name = 'Original Flame Data'
import scipy.io as sio 

X = np.loadtxt('/home/felias/Documentos/Letters_testes/data/Flame_data_original.txt')

#%%
#tamanho do conj. de treino e numero de execuções
vezes = 10
save = True
CP  = 1
IFO = 1
SVM = 1

#CP params:
k_max = 10; outl = 0.0

#IFO params:
cont = 0.07; n_estimators = 75

#OC-SVM params:
nu_ = 0.05


#%%
ii = np.where(X[:,-1] == 1)
ii = ii[0]

Xfi = X[np.where(X[:,-1] == 1)]
Xs = X[np.where(X[:,-1] == 2)]

Xs = np.delete(Xs, np.s_[Xs.shape[1]-1:Xs.shape[1]], axis = 1)
Xfi = np.delete(Xfi, np.s_[Xfi.shape[1]-1:Xfi.shape[1]], axis = 1)

#Determinando tamanho do conjunto de treino:
Nt = int(0.7*Xs.shape[0])

for i in range(5):
    np.random.shuffle(Xs)
#

scaler = StandardScaler()
Xspi = Xs[0:Nt, :]
scaler.fit(Xspi)
Xsp = scaler.transform(copy(Xspi))
Xsti = Xs[Nt:, :]
Xst = scaler.transform(copy(Xsti))
Xst = np.concatenate((Xst, np.ones((Xst.shape[0], 1))), axis = 1)
Xf  = scaler.transform(copy(Xfi))
Xf = np.concatenate((Xf, -1*np.ones((Xf.shape[0], 1))), axis = 1)

# Xsp = Xs[0:Nt, :]
# Xst = Xs[Nt:, :]
# Xst = np.concatenate((Xst, np.ones((Xst.shape[0], 1))), axis = 1)
# Xf = Xfi
# Xf = np.concatenate((Xf, -1*np.ones((Xf.shape[0], 1))), axis = 1)
X_teste = np.concatenate((Xst, Xf), axis = 0)

for i in range(5):
    np.random.shuffle(X_teste)
#


#parametros de desempenho:


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



#%%
#parametros de desempenho:
acc_treino_CP  = []; acc_treino_IFO = []; acc_treino_SVM = []

#modelos:
models_CP = []; models_IFO = []; models_SVM = []

kf = KFold(n_splits = vezes, shuffle=True)

fold = 1 

for train_index, test_index in kf.split(Xsp):
    print('fold = ', fold); fold = fold+1
    X_train, X_test = Xsp[train_index], Xsp[test_index]
    
    if(CP):
        param = kseg.unsupervised_ksegFit(X_train, k_max, 1, 1, outl, 1000)  
        models_CP.append(copy(param))
        saida_val = kseg.unsupervised_ksegPredict(param, X_test)
        
        acc = np.mean(saida_val == np.ones(saida_val.shape))
        acc_treino_CP.append(acc)
    ##
    #Isolation Forest
    if(IFO):
        clf_IFO = IsolationForest(n_estimators = n_estimators, max_samples=50, 
                                  random_state= np.random.RandomState(42), contamination=cont).fit(X_train)
        models_IFO.append(copy(clf_IFO))
        
        saida_val = clf_IFO.predict(X_test)
        acc = np.mean(saida_val == np.ones(saida_val.shape))
        acc_treino_IFO.append(acc)        
    ##
    
     #OC-SVM
    if(SVM):
        clf_SVM = OneClassSVM(gamma='scale', nu = nu_).fit(X_train)
        models_SVM.append(copy(clf_SVM))
        
        saida_val = clf_SVM.predict(X_test)
        acc = np.mean(saida_val == np.ones(saida_val.shape))
        acc_treino_SVM.append(acc)        
        
    ##
if(CP):  del param
if(IFO): del clf_IFO
if(SVM): del clf_SVM 

#%%

if(CP):
    best_cp = np.argmax(acc_treino_CP); param = models_CP[best_cp]
    
    ini = time.time()
    saida_teste = kseg.unsupervised_ksegPredict(param, X_teste[:,0:-1])
    fim = time.time()
    time_CP.append(fim-ini)
    
    fpr, tpr, threshold = calculaROC(param, True, X_teste)
    tprs_CP.append(tpr); fprs_CP.append(fpr); thresholds_CP.append(threshold)   
    
    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste) 
    prec_inl_CP.append(prec_inl); rec_inl_CP.append(rec_inl); f1_inl_CP.append(f1_inl); acc_CP.append(acc)
    prec_out_CP.append(prec_out); rec_out_CP.append(rec_out); f1_out_CP.append(f1_out); f1_CP.append(f1)
##

 #Isolation Forest
if(IFO):
    best_ifo = np.argmax(acc_treino_IFO); clf_IFO = models_IFO[best_ifo]
    
    ini = time.time()
    saida_teste = clf_IFO.predict(X_teste[:,0:-1])
    fim = time.time()
    time_IFO.append(fim-ini)
    fpr, tpr, threshold = calculaROC(clf_IFO, False, X_teste)
    tprs_IFO.append(tpr); fprs_IFO.append(fpr); thresholds_IFO.append(threshold)  

    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste)
    prec_inl_IFO.append(prec_inl); rec_inl_IFO.append(rec_inl); f1_inl_IFO.append(f1_inl); acc_IFO.append(acc)
    prec_out_IFO.append(prec_out); rec_out_IFO.append(rec_out); f1_out_IFO.append(f1_out); f1_IFO.append(f1)
##
 #OC-SVM
if(SVM):
    best_svm = np.argmax(acc_treino_SVM); clf_SVM = models_SVM[best_svm]

    ini = time.time()
    saida_teste = clf_SVM.predict(X_teste[:,0:-1])  
    fim = time.time()
    time_SVM.append(fim-ini)

    fpr, tpr, threshold = calculaROC(clf_SVM, False, X_teste)
    tprs_SVM.append(tpr); fprs_SVM.append(fpr); thresholds_SVM.append(threshold)  

    prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste)
    prec_inl_SVM.append(prec_inl); rec_inl_SVM.append(rec_inl); f1_inl_SVM.append(f1_inl); acc_SVM.append(acc)
    prec_out_SVM.append(prec_out); rec_out_SVM.append(rec_out); f1_out_SVM.append(f1_out); f1_SVM.append(f1)
##

#%%
if(CP == IFO == SVM):
    fprs_total = {'One-Class PC': fprs_CP, 'Isolation Forest': fprs_IFO, 'One-Class SVM': fprs_SVM}
    tprs_total = {'One-Class PC': tprs_CP, 'Isolation Forest': tprs_IFO, 'One-Class SVM': tprs_SVM}
    means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)
    auc_CP =  [means_auc['One-Class PC'], stds_auc['One-Class PC']]
    auc_IFO = [means_auc['Isolation Forest'], stds_auc['Isolation Forest']]
    auc_SVM = [means_auc['One-Class SVM'], stds_auc['One-Class SVM']]
    
elif(CP):
    fprs_total = {'One-Class PC': fprs_CP}
    tprs_total = {'One-Class PC': tprs_CP}
    means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)
    auc_CP =  [means_auc['One-Class PC'], stds_auc['One-Class PC']]

elif(IFO):
    fprs_total = {'Isolation Forest': fprs_IFO}
    tprs_total = {'Isolation Forest': tprs_IFO}
    means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)
    auc_IFO = [means_auc['Isolation Forest'], stds_auc['Isolation Forest']]

elif(SVM):
    fprs_total = {'One-Class SVM': fprs_SVM}
    tprs_total = {'One-Class SVM': tprs_SVM}
    means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)
    auc_SVM = [means_auc['One-Class SVM'], stds_auc['One-Class SVM']]


def exibe(CP, IFO, SVM):
    
    if(CP):
        print("\nOne-Class PC:")
        print('Treino ACC:     ''%.2f' %(np.max(acc_treino_CP)*100))
        print("----- teste ------")
        print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100))
        print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100))
        print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100))
        print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100))
        print('F1-Score Model:    ''%.2f' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100))
        print('ACC Model:         ''%.2f' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100))
        print('AUC Model:         ''%.2f' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100))
        print('Time:              ''%.2f' %((np.mean(time_CP))),'%.2f' %((np.std(time_CP))))
    
    if(IFO):
        print("\nIsolation Forest:")
        print('Treino ACC:     ''%.2f' %(np.max(acc_treino_IFO)*100))
        print("----- teste ------")
        print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_IFO))*100),'%.2f' %((np.std(rec_inl_IFO))*100))
        print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_IFO))*100),'%.2f' %((np.std(rec_out_IFO))*100))
        print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_IFO))*100),'%.2f' %((np.std(f1_inl_IFO))*100))
        print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_IFO))*100),'%.2f' %((np.std(f1_out_IFO))*100))
        print('F1-Score Model:    ''%.2f' %((np.mean(f1_IFO))*100),'%.2f' %((np.std(f1_IFO))*100))
        print('ACC Model:         ''%.2f' %((np.mean(acc_IFO))*100),'%.2f' %((np.std(acc_IFO))*100))
        print('AUC Model:         ''%.2f' %(auc_IFO[0]*100),'%.2f' %(auc_IFO[1]*100))
        print('Time:              ''%.2f' %((np.mean(time_IFO))),'%.2f' %((np.std(time_IFO))))
    
    if(SVM): 
        print("\nOne-Class SVM:")
        print('Treino ACC:     ''%.2f' %(np.max(acc_treino_SVM)*100))
        print("----- teste ------")
        print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_SVM))*100),'%.2f' %((np.std(rec_inl_SVM))*100))
        print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_SVM))*100),'%.2f' %((np.std(rec_out_SVM))*100))
        print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_SVM))*100),'%.2f' %((np.std(f1_inl_SVM))*100))
        print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_SVM))*100),'%.2f' %((np.std(f1_out_SVM))*100))
        print('F1-Score Model:    ''%.2f' %((np.mean(f1_SVM))*100),'%.2f' %((np.std(f1_SVM))*100))
        print('ACC Model:         ''%.2f' %((np.mean(acc_SVM))*100),'%.2f' %((np.std(acc_SVM))*100))
        print('AUC Model:         ''%.2f' %(auc_SVM[0]*100),'%.2f' %(auc_SVM[1]*100))
        print('Time:              ''%.2f' %((np.mean(time_SVM))),'%.2f' %((np.std(time_SVM))))

##

exibe(CP, IFO, SVM)

#%%
if save == True:
    import pickle
    
    file_path = "/home/felias/Documentos/Letters_testes/Resultados/"
    namefile = file_path + "resultado_" + dataset_name + ".txt"
    namefig = file_path + "ROC_" + dataset_name + ".png"
    plt.savefig(namefig)
    
    path_model = "/home/felias/Documentos/Letters_testes/modelos/"
    name_model_CP  = "CP_"  + dataset_name + ".mat"
    name_model_IFO = "IFO_" + dataset_name + ".pkl"
    name_model_SVM = "SVM_" + dataset_name + ".pkl"
    
    sio.savemat(name_model_CP, param)  
    pickle.dump(clf_IFO , open( name_model_IFO , 'wb' ) )
    pickle.dump(clf_SVM , open( name_model_SVM , 'wb' ) ) 
    
    path_data = "/home/felias/Documentos/Letters_testes/data/"
    nome = path_data+'teste_'+dataset_name+'.mat'
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
        
        print("\nIsolation Forest:", file = stream)
        print('Treino ACC:     ''%.2f' %(np.max(acc_treino_IFO)*100), file = stream)
        print("----- teste ------", file = stream)
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
        print('Treino ACC:     ''%.2f' %(np.max(acc_treino_SVM)*100), file = stream)
        print("----- teste ------", file = stream)
        print('Recall InLier:......''%.2f\t' %((np.mean(rec_inl_SVM))*100),'%.2f' %((np.std(rec_inl_SVM))*100), file = stream)
        print('Recall OutLier:.....''%.2f\t' %((np.mean(rec_out_SVM))*100),'%.2f' %((np.std(rec_out_SVM))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_SVM))*100),'%.2f' %((np.std(f1_inl_SVM))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_SVM))*100),'%.2f' %((np.std(f1_out_SVM))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_SVM))*100),'%.2f' %((np.std(f1_SVM))*100), file = stream)
        print('ACC Model...........''%.2f\t' %((np.mean(acc_SVM))*100),'%.2f' %((np.std(acc_SVM))*100), file = stream)
        print('AUC Model...........''%.2f\t' %(auc_SVM[0]*100),'%.2f' %(auc_SVM[1]*100), file = stream)
        print('Time................''%.2f\t' %((np.mean(time_SVM))),'%.2f' %((np.std(time_SVM))), file = stream)
    
        

