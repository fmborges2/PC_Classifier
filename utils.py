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


def findIndeces(X, eps, d, buffer):
    if(X.shape[0] < buffer):
        pwdists = sqdist(X.T, X.T)
        dcp = np.maximum((np.matlib.repmat(d.reshape((d.shape[0], 1)), 1, X.shape[0]) - pwdists), 0)
        vr_size = np.sum(np.minimum(dcp, eps)/eps, axis = 0)
        vr_ns = np.minimum(np.maximum(vr_size,2),3) - 2
        delta = vr_ns*(np.sum(dcp, axis = 0))
        t = np.max(delta)
        i = np.argmax(delta)
        indeces = np.where((dcp[:,i])>0)
        indeces = indeces[0]
        return indeces
    

    else:
        indeces = np.array([])
        vezes = int(X.shape[0]/buffer)
        resto = int( X.shape[0]- vezes*buffer)
        
        for i in range(vezes):
            inicio = int(i*buffer)
            fim = int((i+1)*buffer)
            
            pwdists = sqdist(X[inicio:fim, :].T, X[inicio:fim,:].T)
            dcp = np.maximum((np.matlib.repmat(d[inicio:fim].reshape((buffer, 1)), 1, buffer) - pwdists), 0)
            vr_size = np.sum(np.minimum(dcp[inicio:fim, :], eps)/eps, axis = 0)
            vr_ns = np.minimum(np.maximum(vr_size,2),3) - 2
            delta = vr_ns*(np.sum(dcp, axis = 0))
            t = np.max(delta)
            i = np.argmax(delta)
            ind = np.where((dcp[:,i])>0)
            ind = ind[0]
            indeces = np.append(indeces, ind, axis = 0)
        #end for
        
        if(resto > 0):
            inicio = fim
            fim = X.shape[0]
            
            pwdists = sqdist(X[inicio:fim, :].T, X[inicio:fim,:].T)
            dcp = np.maximum((np.matlib.repmat(d[inicio:fim].reshape((resto, 1)), 1, resto) - pwdists), 0)
            vr_size = np.sum(np.minimum(dcp[inicio:fim, :], eps)/eps, axis = 0)
            vr_ns = np.minimum(np.maximum(vr_size,2),3) - 2
            delta = vr_ns*(np.sum(dcp, axis = 0))
            t = np.max(delta)
            i = np.argmax(delta)
            ind = np.where((dcp[:,i])>0)
            ind = ind[0]
            indeces = np.append(indeces, ind, axis = 0)          
        #end if
        
        return indeces
        

'''funções do kseg'''
''' Fase Operacional -> Mapeamento Distancias'''

def sqdist(a,b):
    aa = np.sum((a**2),axis=0)
    bb = np.sum((b*b),axis=0)
    ab = np.dot((a.T),b)
    c  = aa.reshape((aa.shape[0],1))
    d = np.abs(np.matlib.repmat(c, 1, bb.shape[0])+np.matlib.repmat(bb,aa.shape[0],1) -2*ab)
    return d



def seg_dist(v1, v2, x):
    a = 0
    b = np.linalg.norm(v2-v1)
    u = (v2-v1)/np.linalg.norm(b)
    t = np.dot((x.T - np.matlib.repmat(v1.T, x.shape[1], 1)), u)
    t = np.maximum(t,a)
    t = np.minimum(t,b)
    t = np.reshape(t,(len(t),1))
    u = np.reshape(u,(len(u),1)) 
    p = np.matlib.repmat(v1.T, x.shape[1], 1) + np.dot(t,u.T) 
    d = (x.T - p)
    d = d*d
    d = np.sum(d, axis = 1)
    return d, t, p


'''Fase Desenvolvimento -> Construção e Otimização da CP'''

def hoek(v1, v2):
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    prod = np.dot(v1.T, v2)
    h = np.arccos(prod)/(np.pi)
    return h


def hoek2(X, Y):
    a = np.zeros((2,2))
    a[0,0] = hoek((X[:,0]-X[:,1]), (Y[:,0]-X[:,0])) + hoek((Y[:,0]-X[:,0]), (Y[:,1]-Y[:,0]))
    a[0,1] = hoek((X[:,0]-X[:,1]), (Y[:,2]-X[:,0])) + hoek((Y[:,2]-X[:,0]), (Y[:,3]-Y[:,2]))
    a[1,0] = hoek((X[:,2]-X[:,3]), (Y[:,0]-X[:,2])) + hoek((Y[:,0]-X[:,2]), (Y[:,1]-Y[:,0]))
    a[1,1] = hoek((X[:,2]-X[:,3]), (Y[:,2]-X[:,2])) + hoek((Y[:,2]-X[:,2]), (Y[:,3]-Y[:,2]))
    
    return a


def convertFlip(uu):
    tt = uu.shape
    if len(tt) == 1:
        uu = uu.reshape(1,uu.shape[0], order = 'F')
    
    vv = np.fliplr(uu)
    return vv




def construct_hp(temps, lamda):
    s = np.zeros((temps.shape[0], 4, temps.shape[2]))
    s[:,0:2,:] = temps
    s[:,2,:] = temps[:,1,:]
    s[:,3,:] = temps[:,0,:]
    p = np.zeros((s.shape[2], 2, 2))
    if p.shape[1] == 1:
        p = np.zeros((s.shape[2],2,2))


    p[:,0,0] = np.transpose(np.arange(1, s.shape[2]+1))
    p[:,1,0] = p[:,0,0]
    p[:,0,1] = np.ones((1,s.shape[2]))
    p[:,1,1] = 2*p[:,0,1]
    p = p-1
    p = p.astype(int)
    pl = 2*np.ones((s.shape[2],1))
    pl = pl.astype(int)
    d = np.zeros((s.shape[2], s.shape[2], 4))
    cost = np.zeros(((2*s.shape[2]+1),(2*s.shape[2]+1)))
    
   
    for s1 in range(1, s.shape[2]):
        for s2 in range(s1):
            aa = np.reshape(s[:,0,s1],(s.shape[0], 1))
            bb = np.reshape(s[:,2,s1],(s.shape[0], 1))
            a1 = np.concatenate((aa,bb), axis=1)
            del aa,bb
            aa = np.reshape(s[:,0,s2],(s.shape[0], 1))
            bb = np.reshape(s[:,2,s2],(s.shape[0], 1))
            a2 = np.concatenate((aa,bb), axis=1)
            del aa,bb
            
            l = np.sqrt(sqdist(a1, a2))
            a = hoek2(s[:,:,s1], s[:,:,s2])
            d[s1,s2,:] = np.reshape((l+np.dot(lamda,a)),(1,4), order = 'F')
            d[s2,s1,:] = d[s1,s2,:]
            cost[2*(s1-1)+2:2*(s1-1)+4, 2*(s2-1)+2:2*(s2-1)+4] = l+np.dot(lamda,a)
            cost[2*(s2-1)+2:2*(s2-1)+4, 2*(s1-1)+2:2*(s1-1)+4] = (l+np.dot(lamda,a)).T
    
    #ii = 1
    while (s.shape[2]  > 1):
        #print(ii)
        #ii = ii+1
        mind = np.min(d, axis =2)
        mind = mind + 10000*(np.eye(mind.shape[0]))
        a = np.min(mind, axis = 0)
        s1 = np.argmin(mind, axis = 0)
        s2 = np.argmin(a)
        a = np.min(a)
        teste = s1.shape
        if teste:
            s1 = s1[s2]
        
        i = np.where(d[s1,s2,:] == a)
        i = i[0]
        if (i.shape[0] > 1):
            i = i[0]
        
        if s1 > s2:
            a = s1
            s1 = s2
            s2 = a
        
        if i == 0:
            a = 1
            b = 1
            p[s1,0:pl[s1][0],0] = convertFlip(p[s1,0:pl[s1][0],0])
            p[s1,0:pl[s1][0],1] = convertFlip(p[s1,0:pl[s1][0],1])
            
            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis = 1)
            p = uu
            p[s1,pl[s1][0]:pl[s1][0]+pl[s2][0],:] = p[s2,0:pl[s2][0],:]
            del uu, pp
            
            
        elif i == 1:
            a = 1
            b = 3
            p[s1,0:pl[s1][0],0] = convertFlip(p[s1,0:pl[s1][0],0])
            p[s1,0:pl[s1][0],1] = convertFlip(p[s1,0:pl[s1][0],1])
           
            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis = 1)
            p = uu
            p[s1,pl[s1][0]:pl[s1][0]+pl[s2][0],0] = convertFlip(p[s2,0:pl[s2][0],0])
            p[s1,pl[s1][0]:pl[s1][0]+pl[s2][0],1] = convertFlip(p[s2,0:pl[s2][0],1])
            del uu, pp
                  
           
        elif i==2:
            a=3
            b=1
            
            pp = np.zeros((p.shape[0], 2, 2)) 
            uu = np.append(p, pp, axis = 1)
            p = uu
            p[s1, pl[s1][0]:pl[s1][0]+pl[s2][0],:] = p[s2,0:pl[s2][0],:]
            del uu, pp
            
            
 
        else:
            a=3
            b=3 
            
            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis = 1)
            p = uu
            p[s1, pl[s1][0]:pl[s1][0]+pl[s2][0],0] = convertFlip(p[s2,0:pl[s2][0],0])
            p[s1, pl[s1][0]:pl[s1][0]+pl[s2][0],1] = convertFlip(p[s2,0:pl[s2][0],1])
            del uu, pp
            
        #end ifs 
        p = np.delete(p, np.s_[s2:s2+1], axis=0)
        pl[s1] = pl[s1] + pl[s2]
        pl = np.delete(pl,np.s_[s2], axis = 0)
        
        #create new segment
        i = i+1
        
        if i<3:
            #print('foi 1')
            s[:,0,s1] = s[:,2,s1]
            s[:,1,s1] = s[:,3,s1]
        
        if (np.mod(i,2)):
            #print('foi 2')
            s[:,2,s1] = s[:,2,s2]
            s[:,3,s1] = s[:,3,s2]
        else:
            #print('foi 3')
            s[:,2,s1] = s[:,0,s2]
            s[:,3,s1] = s[:,2,s2]
        
        s = np.delete(s, np.s_[s2:s2+1], axis=2)
        
        d = np.delete(d, np.s_[s2:s2+1], axis=0)
        d = np.delete(d, np.s_[s2:s2+1], axis=1)
        t1 = s1
        
        for t2 in range(s.shape[2]):
#            print('foi')
            if t2<t1:
                s1 = t1
                s2 = t2
            else:
                s1 = t2
                s2 = t1
            if s1!= s2:
                uu1 = np.reshape(s[:,0,s1], (len(s[:,0,s1]),1))
                uu2 = np.reshape(s[:,2,s1], (len(s[:,2,s1]),1))
                uu = np.append(uu1, uu2, axis = 1)
                del uu1, uu2
                
                vv1 = np.reshape(s[:,0,s2], (len(s[:,0,s2]),1))
                vv2 = np.reshape(s[:,2,s2], (len(s[:,2,s2]),1))
                vv = np.append(vv1, vv2, axis = 1)
                del vv1, vv2
                
                l = np.sqrt(sqdist(uu, vv))
                del uu, vv
                aa = hoek(s[:,0,s1]-s[:,1,s1],s[:,1,s2]-s[:,0,s2])
                bb = hoek(s[:,0,s1]-s[:,1,s1],s[:,3,s2]-s[:,2,s2])            
                cc = hoek(s[:,2,s1]-s[:,3,s1],s[:,1,s2]-s[:,0,s2])
                dd = hoek(s[:,2,s1]-s[:,3,s1],s[:,3,s2]-s[:,2,s2])
                a = np.array([[aa,bb],[cc,dd]])
                d[s1,s2,:] = np.reshape((l+lamda*a),(1,4), order = 'F')
                d[s2,s1,:] = d[s1,s2,:];
   
    #end while (s.shape[2]  > 1)
    p = p.astype(int)      
    kernel = 0
    kant = 1
    dummy = 2*temps.shape[2]
    edges = np.zeros((dummy+1,dummy+1))
    
    edges[p[0,0,kant] + 2*(p[0,0,kernel]),dummy] = 1
    edges[dummy,p[0,0,kant]+2*(p[0,0,kernel])] = 1;
    edges[p[0,p.shape[1]-1,kant]+2*(p[0,p.shape[1]-1,kernel]),dummy] = 1;
    edges[dummy,p[0,p.shape[1]-1,kant]+2*(p[0,p.shape[1]-1,kernel])] = 1;

    for knoop in range(pl[0][0]-1):
        edges[p[0,knoop,kant]+2*(p[0,knoop,kernel]),p[0,knoop+1,kant]+2*(p[0,knoop+1,kernel])]= 1+ np.mod(knoop+1,2)
        edges[p[0,knoop+1,kant]+2*(p[0,knoop+1,kernel]),p[0,knoop,kant]+2*(p[0,knoop,kernel])]= 1+ np.mod(knoop+1,2)
    
    return cost, edges



def optim_hp(e,c):
    n = e.shape[0]
    edges = e
    change = 0
    a1 = 0
    a2 = 1
    #teste
#    ii = 1
    
    
    while((a1 < n) or (change)):
#        print('foi')
#        ii = ii+1
        if a1 >= n:
            change = 0
            a1 = 0
            a2 = 1
        
        flipped = 0
        ok = False
        
        while(not ok):
#            print('foi')
            b1 = np.where(edges[:,a1] == 1)
            b1 = b1[0]
            if b1.shape[0] != 0:
                b1 = b1[0]
            
            edges[a1, b1] = 0
            edges[b1, a1] = 0
            b2 = np.where(edges[:,a2] == 1)
            
            if b2[0].shape[0] == 0:
                edges[a1,b1] = 1
                edges[b1,a1] = 1
                a2 = a2+1
                
                if a2>=n:
                    a2 = 0
                    a1 = a1+1
            
            else:
                b2 = b2[0]
                if b2.shape[0] != 0:
                    b2 = b2[0]
            
                edges[a2,b2] = 0
                edges[b2,a2] = 0
                ok = True
        #end while(not ok )
        con = hp_connected(copy(edges),copy(a1),copy(a2))
        
        if(not con):
            temp = a2
            a2 = b2
            b2 = temp
            flipped = 1
        
        uu = c[a1,b2]+c[a2,b1]
        vv = c[a1,b1]+c[a2,b2]
        
        if(uu<vv):
            change = 1
            edges[a1,b2]=1
            edges[b2,a1]=1
            edges[a2,b1]=1
            edges[b1,a2]=1
        
        else:
            edges[a1,b1]=1
            edges[b1,a1]=1
            edges[a2,b2]=1
            edges[b2,a2]=1
            
            if(flipped):
                temp=a2
                a2=b2
                b2=temp
            
            a2 = a2+1
            
            if a2 >= n:
                a2 = 0
                a1 = a1+1
        #end if / else(uu < vv)
    #end while((a1 < n) or (change))
    
    return edges


def  hp_connected(uu, i, j):
    #uu = edges
    
    if i == j:
        con = 1
        return con
    
    while(1):
        nxt = np.where(uu[:,i] > 0)
        
        if nxt[0].shape[0] == 0:
            nxt = nxt[0]
            con = 0
            return con
        
        nxt = nxt[0]
        if nxt.shape[0] != 0:
            nxt = nxt[0]
        uu[nxt, i] = 0
        uu[i, nxt] = 0
        i = nxt
        
        if i==j:
            con = 1
            return con


def map_to_arcl(edges, vertices, x):
    n = x.shape[0]
    D = x.shape[1]
    segments = np.zeros((D, 2, (edges.shape[0] -1)))
    e = edges
    segment = 0
    lengths = np.zeros(((segments.shape[2]+1), 1))
    i = np.where((np.sum(e, axis=0)) ==2);
    i = i[0][0]
    j = np.where((e[i,:]) > 0)
    j = j[0][0]
    
    while segment < (segments.shape[2]):
        e[i,j] = 0
        e[j,i] = 0
        a  = vertices[:,i]
        b  = vertices[:,j]
        a = np.reshape(a,(len(a),1))  
        b = np.reshape(b,(len(b),1))
        c = np.concatenate((a,b),axis=1)
        segments[:,:,segment] = c
        del a,b,c
        lengths[segment + 1] = lengths[segment] + np.linalg.norm(vertices[:,i] - vertices[:,j])
        segment = segment+1
        i = j
        j = np.where((e[i,:])>0)
        if segment < segments.shape[2]:
            j = j[0][0]
    
    y = np.zeros((n, D+1))
    #msqd = 0
    dists = np.zeros((n, segments.shape[2]))
    rest = np.zeros((n, D+1, segments.shape[2]))
   
    for i in range(segments.shape[2]):
        d,t,p = seg_dist(segments[:,0,i], segments[:,1,i], x.T)
        dists[:,i] = d
        a = np.concatenate((p,t), axis=1)
        rest[:,:,i] = a
        del a
    
    d = np.min(dists, axis=1)
    vr = np.argmin(dists, axis = 1)        
    
    for i in range(n):
        y[i,:] = rest[i,:,vr[i]]
        y[i,0] = y[i,0] + lengths[vr[i]]
    
    return y,d
##

def plot_curve(e, v, ax):
    
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