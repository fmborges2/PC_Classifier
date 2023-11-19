# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:05:51 2020

Classificadores baseados em curvas principais 
modelo supervisionado (supervisioned) e 
não-supervisionado (unsupervisioned)


@author: Fernando Elias
"""
import numpy as np
import Kseg_new
from copy import copy

def unsupervised_ksegFit(X, k_max, alfa, lamda, outlier, buffer):
    e,v = Kseg_new.fitCurve(X, k_max, alfa, lamda, buffer)
    y,d = Kseg_new.map_to_arcl(copy(e), copy(v), X)
    d = np.sort(d)[::-1]
    n = len(d)
    
    if outlier == 1:
        dNew = np.array([])
    elif outlier == 0:
        dNew = d
    else:
        inicio = int(np.round(outlier*n))
        dNew = d[inicio-1:]
    
    if dNew.shape[0] == 0:
        limiar = 0
    else:
         limiar = (np.max(dNew) + np.std(dNew))
    #end if 
    
    segments = int(v.shape[1]/2)
    parametros = {'edges': e, 'vertices': v, 'limiar': limiar, 'outlier_rate': outlier,
                  'segments': segments}  
    return parametros


def unsupervised_ksegPredict(parametros, X):
    #função de predição do classificador NÃO SUPERVISIONADO
    #baseado no kseg
    #o valor retornado na saída y é
    #y[i] = 1 o evento é classificado como classe default e 
    #y[i] = -1 o evento é classificado como outlier
    
    e = parametros['edges']
    v = parametros['vertices']
    limiar = parametros['limiar']
    
    uu,d = Kseg_new.map_to_arcl(copy(e), copy(v), X)
    y = np.zeros(len(d))
    
    for i in range(len(d)):
        if(d[i] < limiar):
            y[i] = 1
        else:
            y[i] = -1         
    
    return y


def supervised_ksegFit(X, Y, k_max, alfa, lamda):
    nclasses = Y.shape[1]
    j = 0
    parametros  = {}
    
    while j < nclasses:
        e_nome = 'edges'
        v_nome = 'vertices'
        cont = 0
        for i in range(len(X)):
            if Y[i,j] == 1:
                cont = cont+1
        
        x = np.zeros((cont, X.shape[1]))
        u = 0
        for i in range(len(X)):
            if Y[i,j] == 1:
                x[u, :] = X[i,:]
                u = u+1
        
        j = j+1
        e, v = Kseg_new.fitCurve(x, k_max, alfa, lamda)
        e_nome = e_nome + str(j)
        v_nome = v_nome + str(j)
        parametros[e_nome] = e
        parametros[v_nome] = v
    #end fit
    parametros['nclass'] = nclasses
    return parametros


def supervised_ksegPredict(parametros, X):
    nclasses = parametros['nclass']
    
    d = np.zeros((X.shape[0], nclasses))
    y = np.zeros((X.shape[0], nclasses))
    
    for i in range(nclasses):
        e_nome = 'edges' + str(i+1)
        v_nome = 'vertices' + str(i+1)
        e = parametros[e_nome]
        v = parametros[v_nome]
        
        aux, d[:,i] = Kseg_new.map_to_arcl(copy(e), copy(v), X)
    #end mapping
    uu = np.argmin(d, axis = 1)
    for i in range(len(y)):
        y[i, uu[i]] = 1
    #end predict
    
    return y

def supervised_ksegGetAccuracy(y, yp):
    #calculo da taxa de acerto do classificador
    # y  = valor real do alvo (variavel indicativa de classe)
    # yp = valor predito pelo modelo
    # acc = taxa de acerto referente à cada classe
    
    acc = np.zeros(y.shape[1])
    
    N = np.zeros(y.shape[1]) #número de amostras por cada classe
    for i in range(y.shape[1]):
        a = np.where(y[:,i] == 1)
        a = a[0]
        N[i] = a.shape[0]
    #end for
    
    
    for j in range(y.shape[1]):
        for i in range(y.shape[0]):
            if(y[i, j] == yp[i, j] and y[i, j] == 1):
                acc[j] = acc[j] + 1
            #end if
        #end for
        acc[j] = acc[j]/N[j]
    #end
    
    return acc


            
    

    
        
    
    
    
    
        
        

                
        
                
        
        
    
    
 
    




