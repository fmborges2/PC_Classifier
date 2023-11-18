# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:45:24 2020

@author: Fernando Elias
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:10:04 2019

@author: Fernando Elias
"""

#Função k-seg
import numpy as np
import numpy.matlib
import numpy.linalg as la
from copy import copy
import matplotlib.pyplot as plt
from utils import *

class Kseg:
    def __init__(self, k_max: int, alfa: float, lamda: float, buffer: int):
        self.k_max = k_max
        self.alfa  = alfa
        self.lamda = lamda
        self.buffer = buffer
    #
    
   
    def fitCurve(self, X):
        
        f = 0.5
        
        eps = 2**(-52)
        
        lines = np.zeros((X.shape[1], 2, self.k_max))
        
        k = 1
        
        l,v = la.eigh(np.cov(X.T), 'U') 
        m = np.argmax(l)
        l = np.max(l)               
        cent = np.mean(X, axis =0) 
        
        
        a = cent.T -v[:,m] * f*np.sqrt(l)
        a = a.reshape(a.shape[0],1)
        b = cent.T + v[:,m] * f*np.sqrt(l)
        b = b.reshape(b.shape[0],1)
        c = np.concatenate((a,b), axis=1)
        
        lines[:,:,0] = c
        
        del a,b,c
        
        dists = np.zeros((X.shape[0],k))
        
        for i in range(k):
            d,t,p = seg_dist(lines[:,0,i], lines[:,1,i], X.T)
        
        
        vr = np.ones((X.shape[0], self.k_max))
        dr = np.zeros((self.k_max,1))
        
        cost, edges = construct_hp(lines[:,:,0:k],self.lamda)
        edges = optim_hp(copy(edges), cost)
        
        fim = edges.shape[0]
        edges = np.delete(edges, np.s_[fim-1:fim], axis= 0)
        fim = edges.shape[1]
        edges = np.delete(edges, np.s_[fim-1:fim], axis= 1)
        
        vertices = np.reshape(lines[:,:,0:k], (X.shape[1], 2*k), order='F')
        y, sqd = map_to_arcl(copy(edges), copy(vertices), X)
        
        of = np.array([np.mean(sqd), np.log(np.max(y[:,0]))])
        
        while k<self.k_max:
            
            indeces = findIndeces(X, eps, d, self.buffer)
            indeces = indeces.astype(int)
            
            if indeces.shape[0] < 3:
                print('alocação não mais possível')
                k = self.k_max
                break
            
            XS = np.zeros((indeces.shape[0], X.shape[1]))
            for j in range(indeces.shape[0]):
                XS[j,:] = X[indeces[j], :]
              
            k = k+1    
            
            # print('inserindo segmento ', k)
            
            cent=np.mean(XS, axis = 0)
            
            l,v = la.eigh(np.cov(np.transpose((XS-np.matlib.repmat(cent,XS.shape[0] ,1)))), 'U')
            m = np.argmax(l)
            l = np.max(l)               
            a = cent.T -v[:,m] * f*np.sqrt(l)
            a = a.reshape(a.shape[0],1)
            b = cent.T +v[:,m] * f*np.sqrt(l)
            b = b.reshape(b.shape[0],1)
            c = np.concatenate((a,b), axis=1)
            lines[:,:,k-1] = c
            del a, b, c
            
            change = 1
            
            while (change):
                old_vr = copy(vr)
                #compute voronoi regions and projection distances 
                dists = np.zeros((X.shape[0],k))
                for i in range(k):
                    d,t,p = seg_dist(lines[:,0,i], lines[:,1,i], X.T)
                    dists[:,i] = d
                
                d = np.min(dists, axis = 1)
                vr1 = np.argmin(dists, axis = 1)
                
                for i in range(k):
                    vr[:,i] = np.maximum(-np.abs(vr1-i), -1) +1
                    dr[i] = np.sum(vr[:,i]*dists[:,i])
                
                for i in range(k):
                    if(np.sum(vr[:,i]-old_vr[:,i])) != 0:
                       #print('foi 1')
                       indeces = np.where(vr[:,i] == 1)
                       indeces = indeces[0]
                       XS = np.zeros((indeces.shape[0], X.shape[1]))
                       for j in range(indeces.shape[0]):
                           XS[j,:] = X[indeces[j], :]
                    
                       cent=np.mean(XS, axis = 0)
                       vals, v = la.eigh(np.cov(np.transpose((XS-np.matlib.repmat(cent,XS.shape[0] ,1)))), 'U')
                       l = np.max(vals)
                       m = np.argmax(vals)
                       spread = f*np.sqrt(l)
                       
                       a = cent.T -v[:,m] * spread
                       a = a.reshape(a.shape[0],1)
                       b = cent.T +v[:,m] * spread
                       b = b.reshape(b.shape[0],1)
                       c = np.concatenate((a,b), axis=1)
                       lines[:,:,i] = c
                       del a, b, c
                       
                       d2, t, p = seg_dist(lines[:,0,i], lines[:,1,i], XS.T)
                       
                       if(np.sum(d2) > dr[i]):
                           #print ('foi 2 ')
                           a = lines[:,0,i]+np.min(t)*v[:,m]
                           a = a.reshape(a.shape[0],1)
                           b = lines[:,0,i]+np.max(t)*v[:,m] 
                           b = b.reshape(b.shape[0],1)
                           c = np.concatenate((a,b), axis=1)
                           lines[:,:,i] = c
                           del a,b,c
                
                #end for i in range (k)
                if((old_vr == vr).all()):
                    change = 0    
            #update -> end while(change)
            
            cost, edges = construct_hp(lines[:,:,0:k],self.lamda)
            edges = optim_hp(copy(edges), cost)
        
            fim = edges.shape[0]
            edges = np.delete(edges, np.s_[fim-1:fim], axis= 0)
            fim = edges.shape[1]
            edges = np.delete(edges, np.s_[fim-1:fim], axis= 1)
            
            vertices = np.reshape(lines[:,:,0:k], (X.shape[1], 2*k), order='F')
            y, sqd = map_to_arcl(copy(edges), copy(vertices), X)
            #of = np.append(of, [np.mean(sqd), np.log(np.max(y[:,0]))], axis = 1)
        #end while (k < self.k_max)
        
        self.edges = edges
        self.vertices = vertices
        # print('pronto !')
        return self
    ##
    
    
    def plot_curve(self, ax):
        e = copy(self.edges)
        v = copy(self.vertices)
        
        ws = 5; Cs = 'k'
        wi = 2; Ci = 'k'

        key_s = True
        key_i = True


        for i in range(1, e.shape[0]):
            j = np.where(e[:,i] == 2)[0]
            if j.shape[0] != 0:
                j = j[0]
                if(key_s):
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs, label = 'PC segment')
                    key_s = False
                else:
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs)
            ##
            j = np.where(e[:,i] == 1)[0]
            if j.shape[0] != 0:
                j = j[0]
                if(key_i):
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = wi, color = Ci, label = 'segment connection')
                    key_i = False
                else: 
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = wi, color = Ci)
                ##
            ##
        ##
    ##
                
    


