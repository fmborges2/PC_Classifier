# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:01:04 2021

@author: Fernando Elias
"""
import pickle 
import scipy.io as sio
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
import datetime

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
    
    fig, ax = plt.subplots(dpi = 150)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random Model', alpha=.8)
    
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
            label=r'ROC ' + key,
            lw=2, alpha=.8)
    
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        
        del mean_auc, std_auc
    ##
        
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xlabel("False Positive Rate", fontsize = 12)
    plt.ylabel("True Positive Rate", fontsize = 12)
    ax.legend(loc="lower right")
    plt.show()
    return means, stds
    
##


model_path = "/home/felias/Documentos/Letters_testes/modelos/"

dataset_name = 'annthyroid dataset'
# dataset_name = 'Breast Cancer Wisconsin'
# dataset_name = 'Mammography'
# dataset_name = 'MNIST'
# dataset_name = 'Flame Norm'

path = model_path+ 'CP_' + dataset_name+ '.mat'
param = sio.loadmat(path)

path = model_path+ 'IFO_' + dataset_name+ '.pkl'
clf_IFO = pickle.load(open(path,'rb'))

path = model_path+ 'SVM_' + dataset_name+ '.pkl'
clf_SVM = pickle.load(open(path,'rb'))



data_path = "/home/felias/Documentos/Letters_testes/data/"
path  = data_path + 'teste_'+ dataset_name +'.mat'

data = sio.loadmat(path)

X_teste = data['dados_teste']

save = False

#parametros de desempenho:


tprs_CP = []; fprs_CP = []; thresholds_CP = []
prec_inl_CP = []; rec_inl_CP = []; f1_inl_CP = []; acc_CP = []
prec_out_CP = []; rec_out_CP = []; f1_out_CP = []; f1_CP =  []
time_CP_sample = []; time_CP =[]

tprs_IFO = []; fprs_IFO = []; thresholds_IFO = []
prec_inl_IFO = []; rec_inl_IFO = []; f1_inl_IFO = []; acc_IFO = [] 
prec_out_IFO = []; rec_out_IFO = []; f1_out_IFO = []; f1_IFO  = []
time_IFO_sample = []; time_IFO =[]

tprs_SVM = []; fprs_SVM = []; thresholds_SVM = []
prec_inl_SVM = []; rec_inl_SVM = []; f1_inl_SVM = []; acc_SVM = [] 
prec_out_SVM = []; rec_out_SVM = []; f1_out_SVM = []; f1_SVM  = []
time_SVM_sample = []; time_SVM =[]



#%%
jj = np.random.randint(0, X_teste.shape[0])
sample_test = X_teste[jj,0:-1]; sample_test = sample_test.reshape((1, sample_test.shape[0]))


          
ini = datetime.datetime.now()
uu = kseg.unsupervised_ksegPredict(param, sample_test)
fim = datetime.datetime.now()
tempo = fim-ini
time_CP_sample.append(tempo.microseconds)
del ini, fim, tempo

ini = datetime.datetime.now()
saida_teste = kseg.unsupervised_ksegPredict(param, X_teste[:,0:-1])
fim = datetime.datetime.now()
tempo = fim-ini
time_CP.append(tempo.microseconds)

fpr, tpr, threshold = calculaROC(param, True, X_teste)
tprs_CP.append(tpr); fprs_CP.append(fpr); thresholds_CP.append(threshold)   

prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste) 
prec_inl_CP.append(prec_inl); rec_inl_CP.append(rec_inl); f1_inl_CP.append(f1_inl); acc_CP.append(acc)
prec_out_CP.append(prec_out); rec_out_CP.append(rec_out); f1_out_CP.append(f1_out); f1_CP.append(f1)
##
del ini, fim, tempo
 #Isolation Forest
    
ini = datetime.datetime.now()
uu = clf_IFO.predict(sample_test)
fim = datetime.datetime.now()
tempo = fim-ini
time_IFO_sample.append(tempo.microseconds)
del ini, fim, tempo

ini = datetime.datetime.now()
saida_teste = clf_IFO.predict(X_teste[:,0:-1])
fim = datetime.datetime.now()
tempo = fim-ini
time_IFO.append(tempo.microseconds)

fpr, tpr, threshold = calculaROC(clf_IFO, False, X_teste)
tprs_IFO.append(tpr); fprs_IFO.append(fpr); thresholds_IFO.append(threshold)  

prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste)
prec_inl_IFO.append(prec_inl); rec_inl_IFO.append(rec_inl); f1_inl_IFO.append(f1_inl); acc_IFO.append(acc)
prec_out_IFO.append(prec_out); rec_out_IFO.append(rec_out); f1_out_IFO.append(f1_out); f1_IFO.append(f1)
##
del ini, fim, tempo
 #OC-SVM
 
ini = datetime.datetime.now()
uu = clf_SVM.predict(sample_test)
fim = datetime.datetime.now()
tempo = fim-ini
time_SVM_sample.append(tempo.microseconds)
del ini, fim, tempo

ini = datetime.datetime.now()
saida_teste = clf_SVM.predict(X_teste[:,0:-1])  
fim = datetime.datetime.now()
tempo = fim-ini
time_SVM.append(tempo.microseconds)
    
fpr, tpr, threshold = calculaROC(clf_SVM, False, X_teste)
tprs_SVM.append(tpr); fprs_SVM.append(fpr); thresholds_SVM.append(threshold)  

prec_inl, rec_inl, prec_out, rec_out, f1_inl, f1_out, f1, acc = getResults(X_teste[:,-1], saida_teste)
prec_inl_SVM.append(prec_inl); rec_inl_SVM.append(rec_inl); f1_inl_SVM.append(f1_inl); acc_SVM.append(acc)
prec_out_SVM.append(prec_out); rec_out_SVM.append(rec_out); f1_out_SVM.append(f1_out); f1_SVM.append(f1)
##

#%%

fprs_total = {'One-Class PC': fprs_CP, 'Isolation Forest': fprs_IFO, 'One-Class SVM': fprs_SVM}
tprs_total = {'One-Class PC': tprs_CP, 'Isolation Forest': tprs_IFO, 'One-Class SVM': tprs_SVM}
means_auc, stds_auc = geraROC(tprs_total, fprs_total, dataset_name)
auc_CP =  [means_auc['One-Class PC'], stds_auc['One-Class PC']]
auc_IFO = [means_auc['Isolation Forest'], stds_auc['Isolation Forest']]
auc_SVM = [means_auc['One-Class SVM'], stds_auc['One-Class SVM']]


def exibe(CP, IFO, SVM):
    
    if(CP):
        print("\nOne-Class PC:")
        # print('folds', acc_treino_CP)
        print('Results: ')
        # print('Treino ACC:     ''%.2f' %(np.mean(acc_treino_CP)*100), '%.2f' %((np.std(acc_treino_CP))*100))
        print("----- teste ------")
        print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100))
        print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100))
        print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100))
        print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100))
        print('F1-Score Model:    ''%.2f' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100))
        print('ACC Model:         ''%.2f' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100))
        print('AUC Model:         ''%.2f' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100))
        print('Time sample (us):  ', np.mean(time_CP_sample))
        print('Time test (us):    ', np.mean(time_CP), '   ', np.std(time_CP))
    
    if(IFO):
        print("\nIsolation Forest:")
        # print('folds', acc_treino_IFO)
        print('Results: ')
        # print('Treino ACC:     ''%.2f' %(np.mean(acc_treino_IFO)*100), '%.2f' %((np.std(acc_treino_IFO))*100))
        print("----- teste ------")
        print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_IFO))*100),'%.2f' %((np.std(rec_inl_IFO))*100))
        print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_IFO))*100),'%.2f' %((np.std(rec_out_IFO))*100))
        print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_IFO))*100),'%.2f' %((np.std(f1_inl_IFO))*100))
        print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_IFO))*100),'%.2f' %((np.std(f1_out_IFO))*100))
        print('F1-Score Model:    ''%.2f' %((np.mean(f1_IFO))*100),'%.2f' %((np.std(f1_IFO))*100))
        print('ACC Model:         ''%.2f' %((np.mean(acc_IFO))*100),'%.2f' %((np.std(acc_IFO))*100))
        print('AUC Model:         ''%.2f' %(auc_IFO[0]*100),'%.2f' %(auc_IFO[1]*100))
        print('Time sample (us):  ', np.mean(time_IFO_sample))
        print('Time test (us):    ', np.mean(time_IFO),'    ', np.std(time_IFO))
    
    if(SVM): 
        print("\nOne-Class SVM:")
        # print('folds', acc_treino_SVM)
        print('Results: ')
        # print('Treino ACC:     ''%.2f' %(np.mean(acc_treino_SVM)*100), '%.2f' %((np.std(acc_treino_SVM))*100))
        print("----- teste ------")
        print('Recall InLier:     ''%.2f' %((np.mean(rec_inl_SVM))*100),'%.2f' %((np.std(rec_inl_SVM))*100))
        print('Recall OutLier:    ''%.2f' %((np.mean(rec_out_SVM))*100),'%.2f' %((np.std(rec_out_SVM))*100))
        print('F1-Score InLier:   ''%.2f' %((np.mean(f1_inl_SVM))*100),'%.2f' %((np.std(f1_inl_SVM))*100))
        print('F1-Score OutLier:  ''%.2f' %((np.mean(f1_out_SVM))*100),'%.2f' %((np.std(f1_out_SVM))*100))
        print('F1-Score Model:    ''%.2f' %((np.mean(f1_SVM))*100),'%.2f' %((np.std(f1_SVM))*100))
        print('ACC Model:         ''%.2f' %((np.mean(acc_SVM))*100),'%.2f' %((np.std(acc_SVM))*100))
        print('AUC Model:         ''%.2f' %(auc_SVM[0]*100),'%.2f' %(auc_SVM[1]*100))
        print('Time sample (us):  ', np.mean(time_SVM_sample))
        print('Time test (us):    ''%.2f' %((np.mean(time_SVM))),'%.2f' %((np.std(time_SVM))))

##

exibe(True, True, True)


#%%
if save == True:
    import pickle
    
    file_path  = "C:/Users/Fernando Elias/Google Drive/Artigos_resumos/2020/Electronics Letters/testes/Resultados/"
    model_path = "C:/Users/Fernando Elias/Google Drive/Artigos_resumos/2020/Electronics Letters/testes/modelos/"
    data_path  = "C:/Users/Fernando Elias/Google Drive/Artigos_resumos/2020/Electronics Letters/testes/data/"
    
    namefile = file_path + "resultado_" + dataset_name + ".txt"
    namefig = file_path + "ROC_" + dataset_name + ".png"
    plt.savefig(namefig)
    
    name_model_CP  = model_path + "CP_"  + dataset_name + ".mat"
    name_model_IFO = model_path + "IFO_" + dataset_name + ".pkl"
    name_model_SVM = model_path + "SVM_" + dataset_name + ".pkl"
    name_scaler    = model_path + "Scaler" + dataset_name + ".pkl"
    
    sio.savemat(name_model_CP, param)  
    pickle.dump(clf_IFO , open( name_model_IFO , 'wb' ) )
    pickle.dump(clf_SVM , open( name_model_SVM , 'wb' ) ) 
    # pickle.dump(scaler , open( name_scaler , 'wb' ) ) 
    
    nome = data_path + 'teste_'+dataset_name+'.mat'
    # sio.savemat(nome, {'dados_treino': Xsp, 'dados_teste': X_teste})
       
    with open(namefile, "w") as stream:
        print('Results for dataset: ', dataset_name, file = stream)
        print("\nOne-Class PC:", file = stream)
        # print('folds', acc_treino_CP, file = stream)
        print('Results: ', file = stream)
        # print('Treino ACC:     ''%.2f' %(np.mean(acc_treino_CP)*100), '%.2f' %((np.std(acc_treino_CP))*100), file = stream)
        print("----- teste ------", file = stream)
        print('Recall InLier.......''%.2f\t' %((np.mean(rec_inl_CP))*100),'%.2f' %((np.std(rec_inl_CP))*100), file = stream)
        print('Recall OutLier......''%.2f\t' %((np.mean(rec_out_CP))*100),'%.2f' %((np.std(rec_out_CP))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_CP))*100),'%.2f' %((np.std(f1_inl_CP))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_CP))*100),'%.2f' %((np.std(f1_out_CP))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_CP))*100),'%.2f' %((np.std(f1_CP))*100), file = stream)
        print('ACC Model...........''%.2f\t' %((np.mean(acc_CP))*100),'%.2f' %((np.std(acc_CP))*100), file = stream)
        print('AUC Model...........''%.2f\t' %(auc_CP[0]*100),'%.2f' %(auc_CP[1]*100), file = stream)
        print('Time sample (us)....', np.mean(time_CP_sample), file = stream)
        print('Time test (us)......', np.mean(time_CP), '\t', np.std(time_CP), file = stream)
        print('--------------------------------------------------', file = stream)
        
        print("\nIsolation Forest:", file = stream)
        # print('folds', acc_treino_IFO, file = stream)
        print('Results: ', file = stream)
        # print('Treino ACC:     ''%.2f' %(np.mean(acc_treino_IFO)*100), '%.2f' %((np.std(acc_treino_IFO))*100), file = stream)
        print("----- teste ------", file = stream)
        print('Recall InLier:......''%.2f\t' %((np.mean(rec_inl_IFO))*100),'%.2f' %((np.std(rec_inl_IFO))*100), file = stream)
        print('Recall OutLier:.....''%.2f\t' %((np.mean(rec_out_IFO))*100),'%.2f' %((np.std(rec_out_IFO))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_IFO))*100),'%.2f' %((np.std(f1_inl_IFO))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_IFO))*100),'%.2f' %((np.std(f1_out_IFO))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_IFO))*100),'%.2f' %((np.std(f1_IFO))*100), file = stream)
        print('ACC Model:..........''%.2f\t' %((np.mean(acc_IFO))*100),'%.2f' %((np.std(acc_IFO))*100), file = stream)
        print('AUC Model:..........''%.2f\t' %(auc_IFO[0]*100),'%.2f' %(auc_IFO[1]*100), file = stream)
        print('Time sample (us)....', np.mean(time_IFO_sample), file = stream)
        print('Time test (us)......', np.mean(time_IFO), '\t', np.std(time_IFO), file = stream)
        print('--------------------------------------------------', file = stream)
           
        print("\nOne-Class SVM:", file = stream)
        # print('folds', acc_treino_SVM, file = stream)
        print('Results: ', file = stream)
        # print('Treino ACC:     ''%.2f' %(np.mean(acc_treino_SVM)*100), '%.2f' %((np.std(acc_treino_SVM))*100), file = stream)
        print("----- teste ------", file = stream)
        print('Recall InLier:......''%.2f\t' %((np.mean(rec_inl_SVM))*100),'%.2f' %((np.std(rec_inl_SVM))*100), file = stream)
        print('Recall OutLier:.....''%.2f\t' %((np.mean(rec_out_SVM))*100),'%.2f' %((np.std(rec_out_SVM))*100), file = stream)
        print('F1-Score InLier.....''%.2f\t' %((np.mean(f1_inl_SVM))*100),'%.2f' %((np.std(f1_inl_SVM))*100), file = stream)
        print('F1-Score OutLier....''%.2f\t' %((np.mean(f1_out_SVM))*100),'%.2f' %((np.std(f1_out_SVM))*100), file = stream)
        print('F1-Score Model......''%.2f\t' %((np.mean(f1_SVM))*100),'%.2f' %((np.std(f1_SVM))*100), file = stream)
        print('ACC Model...........''%.2f\t' %((np.mean(acc_SVM))*100),'%.2f' %((np.std(acc_SVM))*100), file = stream)
        print('AUC Model...........''%.2f\t' %(auc_SVM[0]*100),'%.2f' %(auc_SVM[1]*100), file = stream)
        print('Time sample (us)....', np.mean(time_SVM_sample), file = stream)
        print('Time test (us)......', np.mean(time_SVM), '\t', np.std(time_SVM), file = stream)
        


