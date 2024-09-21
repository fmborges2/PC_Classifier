# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:10:04 2019

@author: Fernando Elias
"""

# Função k-seg
import numpy as np
import numpy.linalg as la
from copy import copy
import matplotlib.pyplot as plt


def findIndeces(X: np.ndarray,
                eps: float,
                d: np.ndarray,
                buffer: int):
    if X.shape[0] < buffer:
        pwdists = sqdist(X.T, X.T)
        dcp = np.maximum((np.tile(d.reshape((d.shape[0], 1)), (1, X.shape[0]))- pwdists), 0)
        vr_size = np.sum(np.minimum(dcp, eps) / eps, axis=0)
        vr_ns = np.minimum(np.maximum(vr_size, 2), 3) - 2
        delta = vr_ns * (np.sum(dcp, axis=0))
        i = np.argmax(delta)
        indeces = np.where((dcp[:, i]) > 0)[0]
        return indeces.astype(int)


    else:
        indeces = np.array([])
        vezes = int(X.shape[0] / buffer)
        resto = int(X.shape[0] - vezes * buffer)

        for i in range(vezes):
            inicio = int(i * buffer)
            fim = int((i + 1) * buffer)

            pwdists = sqdist(X[inicio:fim, :].T, X[inicio:fim, :].T)
            dcp = np.maximum((np.tile(d[inicio:fim].reshape((buffer, 1)), (1, buffer)) - pwdists), 0)
            vr_size = np.sum(np.minimum(dcp[inicio:fim, :], eps) / eps, axis=0)
            vr_ns = np.minimum(np.maximum(vr_size, 2), 3) - 2
            delta = vr_ns * (np.sum(dcp, axis=0))
            i = np.argmax(delta)
            ind = np.where((dcp[:, i]) > 0)[0]
            indeces = np.append(indeces, ind, axis=0)

        if resto > 0:
            inicio = fim
            fim = X.shape[0]

            pwdists = sqdist(X[inicio:fim, :].T, X[inicio:fim, :].T)
            dcp = np.maximum((np.tile(d[inicio:fim].reshape((resto, 1)), (1, resto)) - pwdists), 0)
            vr_size = np.sum(np.minimum(dcp[inicio:fim, :], eps) / eps, axis=0)
            vr_ns = np.minimum(np.maximum(vr_size, 2), 3) - 2
            delta = vr_ns * (np.sum(dcp, axis=0))
            i = np.argmax(delta)
            ind = np.where((dcp[:, i]) > 0)[0]
            indeces = np.append(indeces, ind, axis=0)

        return indeces

'''funções do kseg'''
''' Fase Operacional -> Mapeamento Distancias'''


def sqdist(a: np.ndarray, b: np.ndarray):
    aa = np.sum((a ** 2), axis=0)
    bb = np.sum((b * b), axis=0)
    ab = np.dot(a.T, b)
    c = aa.reshape((aa.shape[0], 1))
    d = np.abs(np.tile(c, (1, bb.shape[0])) + np.tile(bb, (aa.shape[0], 1)) - 2 * ab)
    return d


def seg_dist(v1: np.ndarray, v2: np.ndarray, x: np.ndarray):
    a = 0
    b = np.linalg.norm(v2 - v1)
    u = (v2 - v1) / np.linalg.norm(b)
    t = np.dot((x.T - np.tile(v1.T, (x.shape[1], 1))), u)
    t = np.maximum(t, a)
    t = np.minimum(t, b)
    t = np.reshape(t, (len(t), 1))
    u = np.reshape(u, (len(u), 1))
    p = np.tile(v1.T, (x.shape[1], 1)) + np.dot(t, u.T)
    d = (x.T - p)
    d = d * d
    d = np.sum(d, axis=1)
    return d, t, p


'''Fase Desenvolvimento -> Construção e Otimização da CP'''

def hoek(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    prod = np.dot(v1.T, v2)
    h = np.arccos(prod) / np.pi
    return h


def hoek2(X: np.ndarray, Y: np.ndarray):
    a = np.zeros((2, 2))
    a[0, 0] = hoek((X[:, 0] - X[:, 1]), (Y[:, 0] - X[:, 0])) + hoek((Y[:, 0] - X[:, 0]), (Y[:, 1] - Y[:, 0]))
    a[0, 1] = hoek((X[:, 0] - X[:, 1]), (Y[:, 2] - X[:, 0])) + hoek((Y[:, 2] - X[:, 0]), (Y[:, 3] - Y[:, 2]))
    a[1, 0] = hoek((X[:, 2] - X[:, 3]), (Y[:, 0] - X[:, 2])) + hoek((Y[:, 0] - X[:, 2]), (Y[:, 1] - Y[:, 0]))
    a[1, 1] = hoek((X[:, 2] - X[:, 3]), (Y[:, 2] - X[:, 2])) + hoek((Y[:, 2] - X[:, 2]), (Y[:, 3] - Y[:, 2]))

    return a

def convertFlip(uu):

    if len(uu.shape) == 1:
        uu = uu.reshape(1, uu.shape[0], order='F')

    vv = np.fliplr(uu)
    return vv

def construct_hp(temps: np.ndarray, lamda: float):
    s = np.zeros((temps.shape[0], 4, temps.shape[2]))
    s[:, 0:2, :] = temps
    s[:, 2, :] = temps[:, 1, :]
    s[:, 3, :] = temps[:, 0, :]
    p = np.zeros((s.shape[2], 2, 2))

    if p.shape[1] == 1:
        p = np.zeros((s.shape[2], 2, 2))

    p[:, 0, 0] = np.transpose(np.arange(1, s.shape[2] + 1))
    p[:, 1, 0] = p[:, 0, 0]
    p[:, 0, 1] = np.ones((1, s.shape[2]))
    p[:, 1, 1] = 2 * p[:, 0, 1]
    p = p - 1
    p = p.astype(int)
    pl = 2 * np.ones((s.shape[2], 1))
    pl = pl.astype(int)
    d = np.zeros((s.shape[2], s.shape[2], 4))
    cost = np.zeros(((2 * s.shape[2] + 1), (2 * s.shape[2] + 1)))

    for s1 in range(1, s.shape[2]):
        for s2 in range(s1):
            aa = np.reshape(s[:, 0, s1], (s.shape[0], 1))
            bb = np.reshape(s[:, 2, s1], (s.shape[0], 1))
            a1 = np.concatenate((aa, bb), axis=1)
            del aa, bb
            aa = np.reshape(s[:, 0, s2], (s.shape[0], 1))
            bb = np.reshape(s[:, 2, s2], (s.shape[0], 1))
            a2 = np.concatenate((aa, bb), axis=1)
            del aa, bb

            l = np.sqrt(sqdist(a1, a2))
            a = hoek2(s[:, :, s1], s[:, :, s2])
            d[s1, s2, :] = np.reshape((l + np.dot(lamda, a)), (1, 4), order='F')
            d[s2, s1, :] = d[s1, s2, :]
            cost[2 * (s1 - 1) + 2:2 * (s1 - 1) + 4, 2 * (s2 - 1) + 2:2 * (s2 - 1) + 4] = l + np.dot(lamda, a)
            cost[2 * (s2 - 1) + 2:2 * (s2 - 1) + 4, 2 * (s1 - 1) + 2:2 * (s1 - 1) + 4] = (l + np.dot(lamda, a)).T

    while s.shape[2] > 1:

        mind = np.min(d, axis=2)
        mind = mind + 10000 * (np.eye(mind.shape[0]))
        a = np.min(mind, axis=0)
        s1 = np.argmin(mind, axis=0)
        s2 = np.argmin(a)
        a = np.min(a)
        if s1.shape:
            s1 = s1[s2]

        i = np.where(d[s1, s2, :] == a)[0]
        if i.shape[0] > 1:
            i = i[0]

        if s1 > s2:
            a = s1
            s1 = s2
            s2 = a

        if i == 0:
            p[s1, 0:pl[s1][0], 0] = convertFlip(p[s1, 0:pl[s1][0], 0])
            p[s1, 0:pl[s1][0], 1] = convertFlip(p[s1, 0:pl[s1][0], 1])

            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis=1)
            p = uu
            p[s1, pl[s1][0]:pl[s1][0] + pl[s2][0], :] = p[s2, 0:pl[s2][0], :]
            del uu, pp


        elif i == 1:
            p[s1, 0:pl[s1][0], 0] = convertFlip(p[s1, 0:pl[s1][0], 0])
            p[s1, 0:pl[s1][0], 1] = convertFlip(p[s1, 0:pl[s1][0], 1])

            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis=1)
            p = uu
            p[s1, pl[s1][0]:pl[s1][0] + pl[s2][0], 0] = convertFlip(p[s2, 0:pl[s2][0], 0])
            p[s1, pl[s1][0]:pl[s1][0] + pl[s2][0], 1] = convertFlip(p[s2, 0:pl[s2][0], 1])
            del uu, pp

        elif i == 2:

            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis=1)
            p = uu
            p[s1, pl[s1][0]:pl[s1][0] + pl[s2][0], :] = p[s2, 0:pl[s2][0], :]
            del uu, pp

        else:

            pp = np.zeros((p.shape[0], 2, 2))
            uu = np.append(p, pp, axis=1)
            p = uu
            p[s1, pl[s1][0]:pl[s1][0] + pl[s2][0], 0] = convertFlip(p[s2, 0:pl[s2][0], 0])
            p[s1, pl[s1][0]:pl[s1][0] + pl[s2][0], 1] = convertFlip(p[s2, 0:pl[s2][0], 1])
            del uu, pp

        p = np.delete(p, np.s_[s2:s2 + 1], axis=0)
        pl[s1] = pl[s1] + pl[s2]
        pl = np.delete(pl, np.s_[s2], axis=0)

        i += 1

        if i < 3:
            s[:, 0, s1] = s[:, 2, s1]
            s[:, 1, s1] = s[:, 3, s1]

        if np.mod(i, 2):
            s[:, 2, s1] = s[:, 2, s2]
            s[:, 3, s1] = s[:, 3, s2]

        else:
            s[:, 2, s1] = s[:, 0, s2]
            s[:, 3, s1] = s[:, 2, s2]

        s = np.delete(s, np.s_[s2:s2 + 1], axis=2)

        d = np.delete(d, np.s_[s2:s2 + 1], axis=0)
        d = np.delete(d, np.s_[s2:s2 + 1], axis=1)
        t1 = s1

        for t2 in range(s.shape[2]):
            if t2 < t1:
                s1 = t1
                s2 = t2
            else:
                s1 = t2
                s2 = t1
            if s1 != s2:
                uu1 = np.reshape(s[:, 0, s1], (len(s[:, 0, s1]), 1))
                uu2 = np.reshape(s[:, 2, s1], (len(s[:, 2, s1]), 1))
                uu = np.append(uu1, uu2, axis=1)
                del uu1, uu2

                vv1 = np.reshape(s[:, 0, s2], (len(s[:, 0, s2]), 1))
                vv2 = np.reshape(s[:, 2, s2], (len(s[:, 2, s2]), 1))
                vv = np.append(vv1, vv2, axis=1)
                del vv1, vv2

                l = np.sqrt(sqdist(uu, vv))
                del uu, vv
                aa = hoek(s[:, 0, s1] - s[:, 1, s1], s[:, 1, s2] - s[:, 0, s2])
                bb = hoek(s[:, 0, s1] - s[:, 1, s1], s[:, 3, s2] - s[:, 2, s2])
                cc = hoek(s[:, 2, s1] - s[:, 3, s1], s[:, 1, s2] - s[:, 0, s2])
                dd = hoek(s[:, 2, s1] - s[:, 3, s1], s[:, 3, s2] - s[:, 2, s2])
                a = np.array([[aa, bb], [cc, dd]])
                d[s1, s2, :] = np.reshape((l + lamda * a), (1, 4), order='F')
                d[s2, s1, :] = d[s1, s2, :]

    # end while (s.shape[2]  > 1)
    p = p.astype(int)
    kernel = 0
    kant = 1
    dummy = 2 * temps.shape[2]
    edges = np.zeros((dummy + 1, dummy + 1))

    edges[p[0, 0, kant] + 2 * (p[0, 0, kernel]), dummy] = 1
    edges[dummy, p[0, 0, kant] + 2 * (p[0, 0, kernel])] = 1
    edges[p[0, p.shape[1] - 1, kant] + 2 * (p[0, p.shape[1] - 1, kernel]), dummy] = 1
    edges[dummy, p[0, p.shape[1] - 1, kant] + 2 * (p[0, p.shape[1] - 1, kernel])] = 1

    for knoop in range(pl[0][0] - 1):
        edges[p[0, knoop, kant] + 2 * (p[0, knoop, kernel]), p[0, knoop + 1, kant] + 2 * (
        p[0, knoop + 1, kernel])] = 1 + np.mod(knoop + 1, 2)
        edges[p[0, knoop + 1, kant] + 2 * (p[0, knoop + 1, kernel]), p[0, knoop, kant] + 2 * (
        p[0, knoop, kernel])] = 1 + np.mod(knoop + 1, 2)

    return cost, edges


def optim_hp(e: np.ndarray, c: np.ndarray):
    n = e.shape[0]
    edges = e
    change = 0
    a1 = 0
    a2 = 1

    while (a1 < n) or change:

        if a1 >= n:
            change = 0
            a1 = 0
            a2 = 1

        flipped = 0

        while True:
            b1 = np.where(edges[:, a1] == 1)
            b1 = b1[0]
            if b1.shape[0] != 0:
                b1 = b1[0]

            edges[a1, b1] = 0
            edges[b1, a1] = 0
            b2 = np.where(edges[:, a2] == 1)

            if b2[0].shape[0] == 0:
                edges[a1, b1] = 1
                edges[b1, a1] = 1
                a2 = a2 + 1

                if a2 >= n:
                    a2 = 0
                    a1 = a1 + 1

            else:
                b2 = b2[0]
                if b2.shape[0] != 0:
                    b2 = b2[0]

                edges[a2, b2] = 0
                edges[b2, a2] = 0
                # ok = True
                break

        con = hp_connected(copy(edges), copy(a1), copy(a2))

        if not con:
            temp = a2
            a2 = b2
            b2 = temp
            flipped = 1

        uu = c[a1, b2] + c[a2, b1]
        vv = c[a1, b1] + c[a2, b2]

        if uu < vv:
            change = 1
            edges[a1, b2] = 1
            edges[b2, a1] = 1
            edges[a2, b1] = 1
            edges[b1, a2] = 1

        else:
            edges[a1, b1] = 1
            edges[b1, a1] = 1
            edges[a2, b2] = 1
            edges[b2, a2] = 1

            if flipped:
                temp = a2
                a2 = b2
                b2 = temp

            a2 += 1

            if a2 >= n:
                a2 = 0
                a1 += 1

    return edges


def hp_connected(uu: np.ndarray, i: int, j: int):
    if i == j:
        return 1

    while True:
        nxt = np.where(uu[:, i] > 0)

        if nxt[0].shape[0] == 0:
            # nxt = nxt[0]
            return 0

        nxt = nxt[0]
        if nxt.shape[0] != 0:
            nxt = nxt[0]
        uu[nxt, i] = 0
        uu[i, nxt] = 0
        i = nxt

        if i == j:
            return 1


def map_to_arcl(edges: np.ndarray, vertices: np.ndarray, x: np.ndarray):
    n = x.shape[0]
    D = x.shape[1]
    segments = np.zeros((D, 2, (edges.shape[0] - 1)))
    e = edges
    segment = 0
    lengths = np.zeros(((segments.shape[2] + 1), 1))
    i = np.where((np.sum(e, axis=0)) == 2)[0][0]
    j = np.where((e[i, :]) > 0)[0][0]

    while segment < (segments.shape[2]):
        e[i, j] = 0
        e[j, i] = 0
        a = vertices[:, i]
        b = vertices[:, j]
        a = np.reshape(a, (len(a), 1))
        b = np.reshape(b, (len(b), 1))
        c = np.concatenate((a, b), axis=1)
        segments[:, :, segment] = c
        del a, b, c
        lengths[segment + 1] = lengths[segment] + np.linalg.norm(vertices[:, i] - vertices[:, j])
        segment = segment + 1
        i = j
        j = np.where(e[i, :] > 0)
        if segment < segments.shape[2]:
            j = j[0][0]

    y = np.zeros((n, D + 1))
    dists = np.zeros((n, segments.shape[2]))
    rest = np.zeros((n, D + 1, segments.shape[2]))

    for i in range(segments.shape[2]):
        d, t, p = seg_dist(segments[:, 0, i], segments[:, 1, i], x.T)
        dists[:, i] = d
        a = np.concatenate((p, t), axis=1)
        rest[:, :, i] = a
        del a

    d = np.min(dists, axis=1)
    vr = np.argmin(dists, axis=1)

    for i in range(n):
        y[i, :] = rest[i, :, vr[i]]
        y[i, 0] = y[i, 0] + lengths[vr[i]]

    return y, d


def plot_curve(e, v, ax):
    ws = 5
    Cs = 'k'
    wi = 2
    Ci = 'k'

    key_s = True
    key_i = True

    for i in range(1, e.shape[0]):
        j = np.where(e[:, i] == 2)[0]
        if j.shape[0] != 0:
            j = j[0]
            if key_s:
                ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=ws, color=Cs, label='PC segment')
                key_s = False
            else:
                ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=ws, color=Cs)

        j = np.where(e[:, i] == 1)[0]
        if j.shape[0] != 0:
            j = j[0]
            if key_i:
                ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=wi, color=Ci, label='segment connection')
                key_i = False
            else:
                ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=wi, color=Ci)


class Kseg:
    def __init__(self,
                 k_max: int,
                 alfa: float,
                 lamda: float,
                 buffer: int):
        self.k_max = k_max
        self.alfa = alfa
        self.lamda = lamda
        self.buffer = buffer

    def fitCurve(self, X: np.ndarray):
        f = 0.5
        eps = 2 ** (-52)

        lines = np.zeros((X.shape[1], 2, self.k_max))

        l, v = la.eigh(np.cov(X.T), 'U')
        m = np.argmax(l)
        l = np.max(l)
        cent = np.mean(X, axis=0)

        a = cent.T - v[:, m] * f * np.sqrt(l)
        a = a.reshape(a.shape[0], 1)
        b = cent.T + v[:, m] * f * np.sqrt(l)
        b = b.reshape(b.shape[0], 1)
        c = np.concatenate((a, b), axis=1)

        lines[:, :, 0] = c
        del a, b, c

        d, t, p = seg_dist(lines[:, 0, 0], lines[:, 1, 0], X.T)

        cost, edges = construct_hp(lines[:, :, 0:1], self.lamda)
        edges = optim_hp(copy(edges), cost)

        fim = edges.shape[0]
        edges = np.delete(edges, np.s_[fim - 1:fim], axis=0)
        fim = edges.shape[1]
        edges = np.delete(edges, np.s_[fim - 1:fim], axis=1)

        vertices = np.reshape(lines[:, :, 0:1], (X.shape[1], 2), order='F')

        for k in range(self.k_max):

            indeces = findIndeces(X, eps, d, self.buffer)

            if indeces.shape[0] < 3:
                print('alocação não mais possível')
                # k = self.k_max
                break

            XS = np.zeros((indeces.shape[0], X.shape[1]))
            for j in range(indeces.shape[0]):
                XS[j, :] = X[indeces[j], :]

            k += 1

            cent = np.mean(XS, axis=0)

            l, v = la.eigh(np.cov(np.transpose((XS - np.tile(cent, (XS.shape[0], 1))))), 'U')
            m = np.argmax(l)
            l = np.max(l)
            a = cent.T - v[:, m] * f * np.sqrt(l)
            a = a.reshape(a.shape[0], 1)
            b = cent.T + v[:, m] * f * np.sqrt(l)
            b = b.reshape(b.shape[0], 1)
            c = np.concatenate((a, b), axis=1)
            lines[:, :, k - 1] = c
            del a, b, c

            vr = np.ones((X.shape[0], self.k_max))
            dr = np.zeros((self.k_max, 1))
            while True:
                old_vr = copy(vr)
                dists = np.zeros((X.shape[0], k))

                for i in range(k):
                    d, t, p = seg_dist(lines[:, 0, i], lines[:, 1, i], X.T)
                    dists[:, i] = d

                d = np.min(dists, axis=1)
                vr1 = np.argmin(dists, axis=1)

                for i in range(k):
                    vr[:, i] = np.maximum(-np.abs(vr1 - i), -1) + 1
                    dr[i] = np.sum(vr[:, i] * dists[:, i])

                for i in range(k):
                    if (np.sum(vr[:, i] - old_vr[:, i])) != 0:
                        indeces = np.where(vr[:, i] == 1)
                        indeces = indeces[0]
                        XS = np.zeros((indeces.shape[0], X.shape[1]))
                        for j in range(indeces.shape[0]):
                            XS[j, :] = X[indeces[j], :]

                        cent = np.mean(XS, axis=0)
                        vals, v = la.eigh(np.cov(np.transpose((XS - np.tile(cent, (XS.shape[0], 1))))), 'U')
                        l = np.max(vals)
                        m = np.argmax(vals)
                        spread = f * np.sqrt(l)

                        a = cent.T - v[:, m] * spread
                        a = a.reshape(a.shape[0], 1)
                        b = cent.T + v[:, m] * spread
                        b = b.reshape(b.shape[0], 1)
                        c = np.concatenate((a, b), axis=1)
                        lines[:, :, i] = c
                        del a, b, c

                        d2, t, p = seg_dist(lines[:, 0, i], lines[:, 1, i], XS.T)

                        if np.sum(d2) > dr[i]:
                            a = lines[:, 0, i] + np.min(t) * v[:, m]
                            a = a.reshape(a.shape[0], 1)
                            b = lines[:, 0, i] + np.max(t) * v[:, m]
                            b = b.reshape(b.shape[0], 1)
                            c = np.concatenate((a, b), axis=1)
                            lines[:, :, i] = c
                            del a, b, c

                if (old_vr == vr).all():
                    break

            cost, edges = construct_hp(lines[:, :, 0:k], self.lamda)
            edges = optim_hp(copy(edges), cost)

            fim = edges.shape[0]
            edges = np.delete(edges, np.s_[fim - 1:fim], axis=0)
            fim = edges.shape[1]
            edges = np.delete(edges, np.s_[fim - 1:fim], axis=1)

            vertices = np.reshape(lines[:, :, 0:k], (X.shape[1], 2 * k), order='F')

        self.edges = edges
        self.vertices = vertices
        return self

    def plot_curve(self, ax):
        e = copy(self.edges)
        v = copy(self.vertices)

        ws = 5
        Cs = 'k'
        wi = 2
        Ci = 'k'

        key_s = True
        key_i = True

        for i in range(1, e.shape[0]):
            j = np.where(e[:, i] == 2)[0]
            if j.shape[0] != 0:
                j = j[0]
                if key_s:
                    ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=ws, color=Cs, label='PC segment')
                    key_s = False
                else:
                    ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=ws, color=Cs)

            j = np.where(e[:, i] == 1)[0]
            if j.shape[0] != 0:
                j = j[0]
                if key_i:
                    ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=wi, color=Ci, label='segment connection')
                    key_i = False
                else:
                    ax.plot([v[0, i], v[0, j]], [v[1, i], v[1, j]], linewidth=wi, color=Ci)


class OneClassPC:
    def __init__(self,
                 k_max: int,
                 alfa: float,
                 lamda: float,
                 outlier_rate: float,
                 buffer: int):

        self.k_max = k_max
        self.alfa = alfa
        self.lambda_ = lamda
        self.outlier_rate = outlier_rate
        self.buffer = buffer

    def fit(self, X: np.ndarray):
        curve = Kseg(self.k_max, self.alfa, self.lambda_, self.buffer).fitCurve(X)
        y, d = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)
        d = np.sort(d)[::-1]
        n = len(d)

        if self.outlier_rate == 1:
            dNew = np.array([])
        elif self.outlier_rate == 0:
            dNew = d
        else:
            inicio = int(np.round(self.outlier_rate * n))
            dNew = d[inicio - 1:]

        limiar = 0 if dNew.shape[0] == 0 else (np.max(dNew) + np.std(dNew))

        self.segments = int(curve.vertices.shape[1] / 2)
        self.curve = curve
        self.limiar = limiar
        return self


    def predict(self, X: np.ndarray):

        uu, d = map_to_arcl(copy(self.curve.edges), copy(self.curve.vertices), X)
        y = np.zeros(len(d))

        for i in range(len(d)):
            y[i] = 1 if d[i] < self.limiar else -1

        return y


class MultiClassPC:
    def __init__(self, k_max: int = 10,
                 alfa: float = 0.1,
                 lamda: float = 0.5,
                 buffer: int = 200):
        self.k_max = k_max
        self.alfa = alfa
        self.lambda_ = lamda
        self.buffer = buffer

    def fit(self, X: np.ndarray, Y: np.ndarray):
        uu = Y.shape
        if len(uu) == 1:
            self.nclasses = len(np.unique(Y))
            curves = []
            self.class_labels = np.unique(Y)
            self.type_ = '1d'

            for j in range(self.nclasses):
                x = X[Y == self.class_labels[j]]
                curve = Kseg(self.k_max, self.alfa, self.lambda_, self.buffer).fitCurve(x)
                curves.append(curve)
                del curve

            self.curves = curves
            return self

        else:
            self.nclasses = Y.shape[1]
            self.type_ = 'nd'

            curves = []
            for j in range(self.nclasses):
                cont = 0
                for i in range(len(X)):
                    if Y[i, j] == 1:
                        cont = cont + 1

                x = np.zeros((cont, X.shape[1]))
                u = 0
                for i in range(len(X)):
                    if Y[i, j] == 1:
                        x[u, :] = X[i, :]
                        u = u + 1

                curve = Kseg(self.k_max, self.alfa, self.lambda_, self.buffer).fitCurve(x)
                curves.append(curve)
                del curve

            self.curves = curves
            return self

    def predict(self, X: np.ndarray):

        if self.type_ == '1d':
            d = np.zeros((X.shape[0], self.nclasses))
            y = np.zeros(X.shape[0])

            for i, curve in enumerate(self.curves):
                aux, d[:, i] = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)

            uu = np.argmin(d, axis=1)

            for i in range(0, len(self.class_labels)):
                indices = np.where(uu == i)[0]
                y[indices] = self.class_labels[i]
            return y

        else:
            d = np.zeros((X.shape[0], self.nclasses))
            y = np.zeros((X.shape[0], self.nclasses))

            for i, curve in enumerate(self.curves):
                aux, d[:, i] = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)

            uu = np.argmin(d, axis=1)

            for i in range(len(y)):
                y[i, uu[i]] = 1

            return y

    def score(self, y: np.ndarray, yp: np.ndarray):
        # calculo da taxa de acerto do classificador
        # y  = valor real do alvo (variavel indicativa de classe)
        # yp = valor predito pelo modelo
        # acc = taxa de acerto referente à cada classe

        acc = np.zeros(y.shape[1])

        N = np.zeros(y.shape[1])  # número de amostras por cada classe
        for i in range(y.shape[1]):
            a = np.where(y[:, i] == 1)
            a = a[0]
            N[i] = a.shape[0]

        for j in range(y.shape[1]):
            for i in range(y.shape[0]):
                if (y[i, j] == yp[i, j] and y[i, j] == 1):
                    acc[j] = acc[j] + 1

        acc = np.sum(acc) / np.sum(N)
        return np.mean(acc)

    def predict_proba(self, X: np.ndarray):
        d = np.zeros((X.shape[0], self.nclasses))
        probas = np.zeros((X.shape[0], self.nclasses))

        for i, curve in enumerate(self.curves):
            aux, d[:, i] = map_to_arcl(copy(curve.edges), copy(curve.vertices), X)

        for i in range(len(probas)):
            aux = np.sum(1 / (d[i, :]))
            probas[i, :] = (1 / d[i, :]) / aux

        return probas
