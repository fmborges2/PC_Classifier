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
from .utils import *


class Kseg:
    def __init__(self, k_max = 5, alfa = 1, lamda = 1, close = False, buffer = 1000, f = 1.5):
        """
        Class of extraction Principal Curve through the K-segments Algorithm
        parameters: 
            k_max: number of max segments (int, default = 5)
            alfa: parameter of control the length of cuve (float: [0,1], default = 1)
            lamda: control the softness of the curve  (float: [0,1], default = 1)
            buffer: batch size (for less memory consumption) (int, default = 1000)
            closed_curve: indicates if the curve is open or closed (boolean: True- closed curve; False: open curve, default = False)
            f: segment length (float, default 1.5)
        outputs: 
            Class constructed
        """
        
        self.k_max  = k_max
        self.alfa   = alfa
        self.lamda  = lamda
        self.buffer = buffer
        self.close  = close
        self.f      = f
    #end
    
   
    def fitCurve(self, X):
        """
        Function for fit the Principal Curve
        parameters:
            X: input data
        outputs:
            Constructed curve with their parameters
        """
        
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
        """
        Function to mapping the curve and get the euclidean distances,
        associating the data for the segment with less distance
        parameters:
            edges: edges for the principal curve (obtained by Kseg.edges)
            vertices: vertices for the principal curve (obtained by Kseg.vertices)
            x: input data
        outputs:
            y: auxiliar data
            d: euclidean distances between data and curve
        """
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
    
    def plot_curve(self, ax = None, ws = 5, wi = 2, Cs = 'k', Ci='k', marker_ = '_', name_s = 'Segment', name_i = 'Interconnection'):
        """
        Internal Function for plot the Principal Curve
        parameters:
            ax: object from matplotlib.pyplot.subplots for ploting (default = None)
            ws: line width for the segments (int, default = 5)
            wi: line width fot the segmets interconnections (int, default = 2)
            Cs: color of the segments (according of the colors list from matplotlib, default = 'k' (black))
            Ci: color of the interconnections (according of the colors list from matplotlib, default = 'k' (black))
            marker_: plot marker for the curve (from matplotlib.pyplot.plot, according of the list of markers, default = '_') 
            name_s: label for the segment (str, default = 'Segment')
            name_i: label for the interconnection (str, default = 'Interconnection')
        outputs:
            plot of the Principal Curve
        """
        
        e = copy(self.edges)
        v = copy(self.vertices)
        default_plot = False
        
        if ax == None:
            fig, ax = plt.subplots()
            default_plot = True
        #end

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
        if default_plot:
            ax.legend()
    #end
    
    
#end

class OneClassPC:   
    
    def __init__(self, k_max = 5, alfa = 1, lamda = 1, close = False, buffer = 1000, f = 1.5, 
                 outlier_rate = 0.1):
        """
        Constructor of the OneClass PC Classifier 
        parameters: 
            k_max: number of max segments (int, default = 5)
            alfa: parameter of control the length of cuve (float: [0,1], default = 1)
            lamda: control the softness of the curve  (float: [0,1], default = 1)
            buffer: batch size (for less memory consumption) (int, default = 1000)
            closed_curve: indicates if the curve is open or closed (boolean: True- closed curve; False: open curve, default = False)
            f: segment length (float, default 1.5)
            outlier rate: parameter of control quantity of outliers in the data (float: [0,1]| 0: no outliers | 1:full of outliers, default = 0.1)
        outputs: 
            Class constructed
        """
        
        self.k_max = k_max
        self.alfa  = alfa
        self.lamda = lamda
        self.buffer = buffer
        self.close = close
        self.f = f
        self.outlier_rate = outlier_rate
    #end
    
    def fit(self, X):
        """
        Function for training the classifier
        parameters:
            X input train data
        outputs:
            trained model
        """
        
        curve = Kseg(self.k_max, self.alfa, self.lamda, self.close, self.buffer, self.f).fitCurve(X)
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
        """
        Predidct function for the Multi-class PC, using a imput data
        parameters: 
            X: imput data
        outputs: 
            y: predicted target value (Class)
       """
       
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
    def __init__(self, k_max = 5, alfa = 1, lamda = 1, close = False, buffer = 1000, f = 1.5):
        """
        Constructor of the MultiClass PC Classifier 
        parameters: 
            k_max: number of max segments (int, default = 5)
            alfa: parameter of control the length of cuve (float: [0,1], default = 1)
            lamda: control the softness of the curve  (float: [0,1], default = 1)
            buffer: batch size (for less memory consumption) (int, default = 1000)
            closed_curve: indicates if the curve is open or closed (boolean: True- closed curve; False: open curve, default = False)
            f: segment length (float, default 1.5)
        outputs: 
            Class constructed
        """
        
        self.k_max = k_max
        self.alfa  = alfa
        self.lamda = lamda
        self.buffer = buffer
        self.close  = close
        self.f = f
    #end


    def fit(self, X, Y):
        """
        Function for training the classifier
        parameters:
            X input train data
            Y output train data
        outputs:
            trained model
        """
        
        uu = Y.shape
        if len(uu) == 1:
            nclasses = len(np.unique(Y))
            curves = []
            class_values = np.unique(Y)
            self.type_ = '1d'
            
            for j in range(nclasses):                
                x = X[Y == class_values[j]]                
                curve = Kseg(self.k_max, self.alfa, self.lamda, 
                             self.close, self.buffer, self.f).fitCurve(x)
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
        """
        Predidct function for the Multi-class PC, using a imput data
        parameters: 
            X: imput data
        outputs: 
            y: predicted target value (Class)
        """
        
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
        """
        accuracy rate calculus for the classifier 
        parameters: 
            y:  real target value
            yp: predicted target value
        outputs: 
            acc: accuracy score
        """
        
        #calculo da taxa de acerto do classificador
        # y  = valor real do alvo (variavel indicativa de classe)
        # yp = valor predito pelo modelo
        # acc = taxa de acerto referente à cada classe
        
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
        """
        Prediction of the probabilities of each sample (event) to belong each class
        parameters:
            X: the input vectors (n_samples, n_features)
        output: 
            probas: vector containing the probabilities for each class in format (n_samples, n_classes)
        """
        
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