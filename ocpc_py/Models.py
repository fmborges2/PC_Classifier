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
    #Function to use batch processing in the find the distances
    #This method is important to apply to reduces the memory consumption
    #Is used for find indeces for the Voronoi Regions  
    #parameters:
        #X: input data
        #eps: tolerance value
        #d: distances vector
        #buffer: batch size
    #outputs:
        #indeces: Voronoi regions indeces
    
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
#end

'''Kseg Functions'''
'''Operational phase -> distance mapping'''

def sqdist(a,b):
    #Calculate the squared distance between 2 points
    #parameters:
        #a: first point 
        #b: second point
    #outputs:
        #d: squared (euclidean) distance
    
    aa = np.sum((a**2),axis=0)
    bb = np.sum((b*b),axis=0)
    ab = np.dot((a.T),b)
    c  = aa.reshape((aa.shape[0],1))
    d = np.abs(np.matlib.repmat(c, 1, bb.shape[0])+np.matlib.repmat(bb,aa.shape[0],1) -2*ab)
    return d
#end


def seg_dist(v1, v2, x):
    #Calculate the distance between a segment and data
    #parameters:
        #v1: start vertex
        #v2: end vertex
        #x: input data
    #outputs:
        #d: distances between segment and data
        #t: projection index
        #p: projected data points on the curve
    
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
#end

'''Fase Desenvolvimento -> Construção e Otimização da CP'''
'''Development curve phase -> Construction and optimization of PC'''

def hoek(v1, v2):
    #compute angle between 2 vertices
    #parameters:
        #v1: first vertex
        #v2: second vertex
    #outputs:
        #h: angle between the vertices
    
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    prod = np.dot(v1.T, v2)
    h = np.arccos(prod)/(np.pi)
    return h
#end

def hoek2(X, Y):
    #compute angle between segments X & S and S & Y
    #where S is the segment connecting X and Y 
    #both X and Y are 2x4 matrices with endpoints in columns
    
    a = np.zeros((2,2))
    a[0,0] = hoek((X[:,0]-X[:,1]), (Y[:,0]-X[:,0])) + hoek((Y[:,0]-X[:,0]), (Y[:,1]-Y[:,0]))
    a[0,1] = hoek((X[:,0]-X[:,1]), (Y[:,2]-X[:,0])) + hoek((Y[:,2]-X[:,0]), (Y[:,3]-Y[:,2]))
    a[1,0] = hoek((X[:,2]-X[:,3]), (Y[:,0]-X[:,2])) + hoek((Y[:,0]-X[:,2]), (Y[:,1]-Y[:,0]))
    a[1,1] = hoek((X[:,2]-X[:,3]), (Y[:,2]-X[:,2])) + hoek((Y[:,2]-X[:,2]), (Y[:,3]-Y[:,2]))
    
    return a
#end

def convertFlip(uu):
    #adaptation for the original matlab algorithm, to incoporate the same vector structure
    
    tt = uu.shape
    if len(tt) == 1:
        uu = uu.reshape(1,uu.shape[0], order = 'F')
    
    vv = np.fliplr(uu)
    return vv
#end



def construct_hp(temps, lamda):
    #Function to construt new segment and reorganize the segments in the edges matrix 
    #parameters:
        #temps: current segments in the PC
        #lamda: parameter of softness
    #outputs:
        #edges: matrix that indicates the vertices connection
        #costs: loss for optimization
    
    
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
#end


def optim_hp(e,c):
    #optimizitation of the curve, of the connections of the segments
    #parameters: 
        #e: edges matrix
        #c: cost value from the contruct_hp
    #outputs:
        #edges: optimized edges matrix
    
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
#end

def  hp_connected(uu, i, j):
    #uu = edges
    #verify if 2 vertices are connected
    
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
#end

def map_to_arcl(edges, vertices, x):
    #Function to mapping the curve and get the euclidean distances,
    #associating the data for the segment with less distance
    #parameters:
        #edges: edges for the principal curve (obtained by Kseg.edges)
        #vertices: vertices for the principal curve (obtained by Kseg.vertices)
        #x: input data
    #outputs:
        #y: auxiliar data
        #d: euclidean distances between data and curve
    #WARNING: always pass the copy of the edges and vertices to avoid modifications on the main variable.
    
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
#end

def plot_curve(e, v, ax, ws, wi, Cs, Ci, marker_, name_s, name_i):
    #External Function for plot the Principal Curve
    #parameters:
        #e: edges for the principal curve (obtained by Kseg.edges)
        #v: vertices for the principal curve (obtained by Kseg.vertices)
        #ax: object from matplotlib.pyplot.subplots for ploting
        #ws: line width for the segments 
        #wi: line width fot the segmets interconnections
        #Cs: color of the segments
        #Ci: color of the interconnections
        #marker_: plot marker for the curve (from matplotlib.pyplot.plot)
        #name_s: label for the segment
        #name_i: label for the interconnection
    #outputs:
        #plot of the Principal Curve
    
    
    # Cs = 'k'; Ci = 'k'
    # ws = 5; wi = 2

    key_s = True
    key_i = True

    for i in range(1, e.shape[0]):
        j = np.where(e[:,i] == 2)[0]
        if j.shape[0] != 0:
            j = j[0]
            if(key_s):
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]],  color = Cs, marker = marker_, label = name_s)
                key_s = False
            else:
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]],  color = Cs, marker = marker_)
        ##
        j = np.where(e[:,i] == 1)[0]
        if j.shape[0] != 0:
            j = j[0]
            if(key_i):
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]],  color = Ci, marker = marker_, label = name_i)
                key_i = False
            else: 
                ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]],  color = Ci, marker = marker_)
#end



class Kseg:
    def __init__(self, k_max: int, alfa: float, lamda: float, close: bool, buffer: int, f: float):
        #Class of extraction Principal Curve through the K-segments Algorithm
        #parameters: 
            #k_max: number of max segments (int)
            #alfa: parameter of control the length of cuve (float: [0,1])
            #lamda: control the softness of the curve  (float: [0,1])
            #buffer: batch size (for less memory consumption) (int)
            #closed_curve: indicates if the curve is open or closed (boolean: True- closed curve; False: open curve)
            #f: segment length (default 1.5)
        #outputs: 
            #Class constructed
        
        self.k_max  = k_max
        self.alfa   = alfa
        self.lamda  = lamda
        self.buffer = buffer
        self.close  = close
        self.f      = f
    #end
    
   
    def fitCurve(self, X):
        #Function for fit the Principal Curve
        #parameters:
            #X: input data
        #outputs:
            #Constructed curve with their parameters
        
        f = copy(self.f)
        
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
        
        cost, edges = construct_hp(lines[:,:,0:k], self.lamda)
        edges = optim_hp(copy(edges), cost)
        
        fim = edges.shape[0]
        edges = np.delete(edges, np.s_[fim-1:fim], axis= 0)
        fim = edges.shape[1]
        edges = np.delete(edges, np.s_[fim-1:fim], axis= 1)
        
        vertices = np.reshape(lines[:,:,0:k], (X.shape[1], 2*k), order='F')
        y, sqd = map_to_arcl(copy(edges), copy(vertices), X)
        
        # of = np.array([np.mean(sqd), np.log(np.max(y[:,0]))])
        
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
        
        if self.close:
            con = []
            for k in range(self.edges.shape[1]):
                u = np.where(self.edges[:,k] == 1)
                if len(u[0]) == 0:
                    con.append(k)
                #
            #
            self.edges[con[0], con[1]] = 1
            self.edges[con[1], con[0]] = 1
        #
        
        return self
    #end
    
    def map_to_arcl(self, x):
        #Function to mapping the curve and get the euclidean distances,
        #associating the data for the segment with less distance
        #parameters:
            #edges: edges for the principal curve (obtained by Kseg.edges)
            #vertices: vertices for the principal curve (obtained by Kseg.vertices)
            #x: input data
        #outputs:
            #y: auxiliar data
            #d: euclidean distances between data and curve
        #WARNING: always pass the copy of the edges and vertices to avoid modifications on the main variable.
        
        edges = copy(self.edges)
        vertices = copy(self.vertices)
        
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
    #end
    
    def plot_curve(self, ax, ws, wi, Cs, Ci, marker_, name_s, name_i):
        #Internal Function for plot the Principal Curve
        #parameters:
            #e: edges for the principal curve (obtained by Kseg.edges)
            #v: vertices for the principal curve (obtained by Kseg.vertices)
            #ax: object from matplotlib.pyplot.subplots for ploting
            #ws: line width for the segments 
            #wi: line width fot the segmets interconnections
            #Cs: color of the segments
            #Ci: color of the interconnections
            #marker_: plot marker for the curve (from matplotlib.pyplot.plot)
            #name_s: label for the segment
            #name_i: label for the interconnection
        #outputs:
            #plot of the Principal Curve
        
        e = copy(self.edges)
        v = copy(self.vertices)

        key_s = True
        key_i = True

        for i in range(1, e.shape[0]):
            j = np.where(e[:,i] == 2)[0]
            if j.shape[0] != 0:
                j = j[0]
                if(key_s):
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs, marker = marker_, label = name_s)
                    key_s = False
                else:
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]], linewidth = ws, color = Cs, marker = marker_)
            ##
            j = np.where(e[:,i] == 1)[0]
            if j.shape[0] != 0:
                j = j[0]
                if(key_i):
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]],  linewidth = wi, color = Ci, marker = marker_, label = name_i)
                    key_i = False
                else: 
                    ax.plot([v[0,i], v[0,j]], [v[1,i], v[1,j]],  linewidth = wi, color = Ci, marker = marker_)
    #end
    
    
#end

class OneClassPC:   
    
    def __init__(self, k_max: int, alfa: float, lamda: float, buffer: int, closed_curve: bool, f: float, 
                 outlier_rate: float):    
        #Constructor of the OneClass PC Classifier 
        #parameters: 
            #k_max: number of max segments (int)
            #alfa: parameter of control the length of cuve (float: [0,1])
            #lamda: control the softness of the curve  (float: [0,1])
            #buffer: batch size (for less memory consumption) (int)
            #closed_curve: indicates if the curve is open or closed (boolean: True- closed curve; False: open curve)
            #f: segment length (default 1.5)
            #outlier rate: parameter of control quantity of outliers in the data (float: [0,1]| 0: no outliers | 1:full of outliers)
        #outputs: 
            #Class constructed
        
        self.k_max = k_max
        self.alfa  = alfa
        self.lamda = lamda
        self.buffer = buffer
        self.closed_curve = closed_curve
        self.f = f
        self.outlier_rate = outlier_rate
    #end
    
    def fit(self, X):
        #Function for training the classifier
        #parameters:
            #X input train data
        #outputs:
            #trained model
        curve = Kseg(self.k_max, self.alfa, self.lamda, self.closed_curve, self.buffer, self.f).fitCurve(X)
        y,d = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)
        d = np.sort(d)[::-1]
        n = len(d)
        
        if self.outlier_rate == 1:
            dNew = np.array([])
        elif self.outlier_rate == 0:
            dNew = d
        else:
            inicio = int(np.round(self.outlier_rate*n))
            dNew = d[inicio-1:]
        
        if dNew.shape[0] == 0:
            limiar = 0
        else:
             limiar = (np.max(dNew) + np.std(dNew))
        #end if 
        
        segments = int(curve.vertices.shape[1]/2)
        
        self.curve = curve
        self.limiar = limiar
        self.segments = segments
        return self
    #end
    
    def predict(self, X):
        #Predidct function for the Multi-class PC, using a imput data
        #parameters: 
            # X: imput data
        #outputs: 
            # y: predicted target value (Class)
       
        uu,d = map_to_arcl(copy(self.curve.edges), copy(self.curve.vertices), X)
        y = np.zeros(len(d))
        
        for i in range(len(d)):
            if(d[i] < self.limiar):
                y[i] = 1
            else:
                y[i] = -1         
        
        return y
    #end
#end


class MultiClassPC:
    def __init__(self, k_max: int, alfa: float, lamda: float, buffer: int,
                 closed_curve: bool, f: float):
        
        #Constructor of the MultiClass PC Classifier 
        #parameters: 
            #k_max: number of max segments (int)
            #alfa: parameter of control the length of cuve (float: [0,1])
            #lamda: control the softness of the curve  (float: [0,1])
            #buffer: batch size (for less memory consumption) (int)
            #closed_curve: indicates if the curve is open or closed (boolean: True- closed curve; False: open curve)
            #f: segment length (default 1.5)
        #outputs: 
            #Class constructed
        
        self.k_max = k_max
        self.alfa  = alfa
        self.lamda = lamda
        self.buffer = buffer
        self.closed_curve = closed_curve
        self.f = f
    #end


    def fit(self, X, Y):
        #Function for training the classifier
        #parameters:
            #X input train data
            #Y output train data
        #outputs:
            #trained model
        
        uu = Y.shape
        if len(uu) == 1:
            nclasses = len(np.unique(Y))
            curves = []
            class_values = np.unique(Y)
            self.type_ = '1d'
            
            for j in range(nclasses):                
                x = X[Y == class_values[j]]                
                curve = Kseg(self.k_max, self.alfa, self.lamda, 
                             self.closed_curve, self.buffer, self.f).fitCurve(x)
                # e_nome = e_nome + str(j)
                # v_nome = v_nome + str(j)
                # parametros[e_nome] = e
                # parametros[v_nome] = v
                curves.append(curve)
                del curve
            #end fit
            self.class_labels = class_values
            self.nclasses = nclasses
            self.curves = curves
            return self
            
        #
        else:
            nclasses = Y.shape[1]
            self.type_ = 'nd'
            
            j = 0
            curves = []
            while j < nclasses:
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
                curve = Kseg(self.k_max, self.alfa, self.lamda, self.buffer).fitCurve(x)
                # e_nome = e_nome + str(j)
                # v_nome = v_nome + str(j)
                # parametros[e_nome] = e
                # parametros[v_nome] = v
                curves.append(curve)
                del curve
            #end fit
            self.nclasses = nclasses
            self.curves = curves
            return self
    #end
    
    def predict(self, X):
        #Predidct function for the Multi-class PC, using a imput data
        #parameters: 
            # X: imput data
        #outputs: 
            # y: predicted target value (Class)
        
        if self.type_ == '1d':
            d = np.zeros((X.shape[0], self.nclasses))
            y = np.zeros(X.shape[0])
            i = 0
            for curve in self.curves:          
                aux, d[:,i] = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)
                i+=1
            #end mapping
            
            uu = np.argmin(d, axis = 1)
            
            for i in range(0, len(self.class_labels)):
                indices = np.where(uu == i)[0]
                y[indices] = self.class_labels[i]
            #
            return y
        #
        else:
            d = np.zeros((X.shape[0], self.nclasses))
            y = np.zeros((X.shape[0], self.nclasses))
            i = 0
            for curve in self.curves:          
                aux, d[:,i] = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)
                i+=1
            #end mapping
            uu = np.argmin(d, axis = 1)
            
            for i in range(len(y)):
                y[i, uu[i]] = 1
            #end predict
            
            return y
    #end
    
    def score(self, y, yp):
        #calculo da taxa de acerto do classificador
        # y  = valor real do alvo (variavel indicativa de classe)
        # yp = valor predito pelo modelo
        # acc = taxa de acerto referente à cada classe
        
        #accuracy rate calculus for the classifier 
        #parameters: 
            # y:  real target value
            # yp: predicted target value
        #outputs: 
            # acc: accuracy score
        
        acc = np.zeros(y.shape[0])
        
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
        #end
        acc = np.sum(acc)/np.sum(N)
        return np.mean(acc)
    #end
    
    def predict_proba(self, X):
        # Prediction of the probabilities of each sample (event) to belong each class
        # parameters:
            # X: the input vectors (n_samples, n_features)
        #output: 
            # probas: vector containing the probabilities for each class in format (n_samples, n_classes)
        
        
        d = np.zeros((X.shape[0], self.nclasses))
        probas = np.zeros((X.shape[0], self.nclasses))
        
        i = 0
        for curve in self.curves:          
            aux, d[:,i] = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)
            i+=1
        ##
        
        for i in range(len(probas)):
            aux = np.sum(1/(d[i,:]))
            # probas[i, :] = (d[i, :] / total_dists[i])
            probas[i,:] = (1/d[i,:]) / aux
            # print('----------', i)
            # print(d[i, :])
            # print(aux)
            # print(probas[i,:])
            # print(np.sum(probas[i,:]))
        #end predict
        
        return probas
    #end       
#end