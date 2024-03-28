import memory_profiler
from scipy.interpolate import pchip_interpolate
from functools import partial
from p_tqdm import p_map
from scipy.stats import skew
import random
import pickle
import os
from tqdm import tqdm
import csv
from collections import defaultdict
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import operator
import contextlib
import functools
import operator
import warnings
from numpy.core import overrides
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def doanes_formula(data, n):
    '''
    Functions for computing number of bins for data using Doane's formula
    data : input array
    n : number of elements in data
    '''

    g1 = skew(data)
    sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))

    k = 1 + np.log2(n) + np.log2(1 + np.abs(g1) / sigma_g1)

    for i in range(len(k)):
        k[i] = int(np.ceil((2 / 3) * k[i]))  # Round up to the nearest integer

    return k


def binscalc(X, n, d, method):
    '''
    Function to calculate number of bins in a multidimensional histogram using a generalised Freedman–Diaconis rule.

    Parameters
    ----------
    x   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    Returns
    -------

    Bins : array
         number of bins along each dimension

    References
    ----------
    - Freedman, David, and Persi Diaconis. "On the histogram as a density estimator: L 2 theory." Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete 57.4 (1981): 453-476.
    - Birgé, Lucien, and Yves Rozenholc. "How many bins should be put in a regular histogram." ESAIM: Probability and Statistics 10 (2006): 24-45.

    '''
    if method == 'FD':  # generalised Freedman–Diaconis rule.
        # Generalised Freedman-Diaconis rule with Bins[i] \propto n^(1/(d+2))
        mult_fact = (n**(1 / (2 + d + 10**(-9))))
        Bins = np.zeros(d)
        inf_arr = np.amin(X, axis=0)
        sup_arr = np.amax(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        IQR = (q75 - q25) + 10**(-9)
        print('IQR:', IQR)
        Bins_1 = np.ceil(mult_fact * (sup_arr - inf_arr) / (2 * IQR))
        Bins = Bins_1.astype(int)
    elif method == 'sqrt':
        ns = [n] * d
        Bins_1 = np.ceil(np.sqrt(ns))
        Bins = Bins_1.astype(int)

    elif method == 'rice':
        ns = [n] * d
        Bins_1 = 2 * np.ceil(np.cbrt(ns))
        Bins = Bins_1.astype(int)
    elif method == 'sturges':
        ns = [n] * d
        Bins_1 = 1 + np.ceil(np.log(ns))
        Bins = Bins_1.astype(int)
    elif method == 'doanes':
        Bins_1 = doanes_formula(X, n)
        Bins = Bins_1.astype(int)

    elif method == 'default':
        Bins = 15

    # print('inf:',inf)
    # print('sup:',sup)
    # print('IQR:',iqr)

    return Bins


def mutualinfo(X, Y, n, d):
    '''
    Function to calculate mutual information between to time series

    Parameters
    ----------
    x   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    y   : ndarray
        double array of shape (n,d).  second time series

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    Returns
    -------

    MI : double
         mutual information between time series

    References
    ----------
    - Shannon, Claude Elwood. "A mathematical theory of communication." The Bell system technical journal 27.3 (1948): 379-423.

    '''

    points = np.concatenate((X, Y), axis=1)
    bins = binscalc(points, n, 2 * d, 'FD')
    print('BINS:', bins)
    # print('bins:',bins.shape)
    # print('points:',points.shape)
    # 10^-9 added so that x log x does not diverge when x=0 in the calculation
    # of mutual information
    p_xy = np.histogramdd(points, bins=binscalc(
        points, n, 2 * d, 'default'))[0] + 10**(-9)
    p_x = np.histogramdd(X, bins=binscalc(X, n, d, 'default'))[0] + 10**(-9)
    p_y = np.histogramdd(Y, bins=binscalc(Y, n, d, 'default'))[0] + 10**(-9)
    p_xy /= np.sum(p_xy)  # Normalising the probability distribution
    p_x /= np.sum(p_x)
    p_y /= np.sum(p_y)
    return np.sum(p_xy * np.log2(p_xy)) - np.sum(p_x * np.log2(p_x)) - \
        np.sum(p_y * np.log2(p_y))  # formula for mutual information


def timedelayMI(u, n, d, tau):
    '''
    Function to calculate mutual information between a time series and a delayed version of itself

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
        amount of delay

    Returns
    -------

    MI : double
         mutual information between u and u delayed by tau

    References
    ----------
    - Shannon, Claude Elwood. "A mathematical theory of communication." The Bell system technical journal 27.3 (1948): 379-423.

    '''

    X = u[0:n - tau, :]
    Y = u[tau:n, :]
    return mutualinfo(X, Y, n - tau, d)


def findtau(u, n, d, grp):
    '''
    Function to calculate correct delay for estimating embedding dimension based on the first minima of the tau vs mutual information curve

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    Returns
    -------

    tau : double
         optimal amount of delay for which mutual information reaches its first minima(and global minima, in case first minima doesn't exist)

    '''

    TAU = []
    MIARR = []
    minMI = timedelayMI(u, n, d, 1)
    for tau in range(2, n):
        nextMI = timedelayMI(u, n, d, tau)
        TAU.append(tau)
        MIARR.append(nextMI)
        if nextMI > minMI:
            break
        minMI = nextMI

    return tau - 1

#### Calculation of m ####################################################


def delayseries(u, n, d, tau, m):
    '''
    Function to calculate correct delay for estimating embedding dimension based on the first minima of the tau vs mutual information curve

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
         amount of delay

    m    : int
         number of embedding dimensions

    Returns
    -------

    s : ndarray
       3D matrix ov size ((n-(m-1)*tau,m,d)), time delayed embedded signal

    References
    ----------
    - Takens, F. (1981). Dynamical systems and turbulence. Warwick, 1980, 366–381.

    '''
    s = np.zeros((n - (m - 1) * tau, m, d))
    for i in range(n - (m - 1) * tau):
        for j in range(m):
            s[i, j] = u[i + j * tau]
    return s


def nearest(s, n, d, tau, m):
    '''
    Function that would give a nearest neighbour map, the output array(nn) stores indices of nearest neighbours for index values of each observations

    Parameters
    ----------
    s   : ndarray
        3D matrix ov size ((n-(m-1)*tau,m,d)), time delayed embedded signal

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
         amount of delay

    m    : int
         number of embedding dimensions

    Returns
    -------

    nn : array
       an array denoting index of nearest neighbour for each observation

    References
    ----------
    - Takens, F. (1981). Dynamical systems and turbulence. Warwick, 1980, 366–381.

    '''
    nn = np.zeros(n - m * tau, dtype=int)
    nn[0] = n - m * tau - 1
    for i in range(n - m * tau):
        for j in range(n - m * tau):
            if (i != j and np.linalg.norm(
                    s[i] - s[j]) < np.linalg.norm(s[i] - s[nn[i]])):
                nn[i] = j
    return nn


def fnnratio(u, n, d, m, tau, r, sig):
    '''
    Function that calculates the ratio of false nearest neighbours

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
         amount of delay

    m    : int
         number of embedding dimensions

    r    : double
         ratio parameter

    sig : double
         standard deviation of the data

    Returns
    -------

    FNN : double
       ratio of false nearest neighbours, for a given embedding dimension m, when compared an embedding dimension m+1

    References
    ----------
    - Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45 (6), 3403.

    '''
    s1 = delayseries(u, n, d, tau, m)     # embedding in m dimensions
    s2 = delayseries(u, n, d, tau, m + 1)   # embedding in m+1 dimensions
    # containg nearest neghbours after embedding in m dimensions
    nn = nearest(s1, n, d, tau, m)
    isneigh = np.zeros(n - m * tau)
    isfalse = np.zeros(n - m * tau)
    for i in range(n - m * tau):
        disto = np.linalg.norm(s1[i] - s1[nn[i]]) + 10**(-9)
        distp = np.linalg.norm(s2[i] - s2[nn[i]])
        if (disto < sig / r):
            isneigh[i] = 1
            if (distp / disto > r):
                isfalse[i] = 1
    return sum(isneigh * isfalse) / (sum(isneigh) + 10**(-9))


def fnnhitszero(u, n, d, m, tau, sig, delta, Rmin, Rmax, rdiv):
    '''
    Function that finds the value of r at which the FNN ratio can be effectively considered as zero

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
         amount of delay

    m    : int
         number of embedding dimensions

    r    : double
         ratio parameter

    sig : double
         standard deviation of the data

    delta: double
         the tolerance value(the maximum difference from zero) for a value of FNN ratio to be effectively considered to be zero

    Rmin : double
         minimum value of r from where we would start the parameter search

    Rmax : double
         maximum value of r for defining the upper limit of parameter search

    rdiv : Int
         number of divisions between Rmin and Rmax for parameter search

    Returns
    -------

    r : double
       value of r at which the value of FNN ratio effectively hits zero

    References
    ----------
    - Kantz, H., & Schreiber, T. (2004). Nonlinear time series analysis (Vol. 7). Cambridge university press. section 3.3.1

    '''
    Rarr = np.linspace(Rmin, Rmax, rdiv)
    for i in range(rdiv):
        if fnnratio(u, n, d, m, tau, Rarr[i], sig) < delta:
            return Rarr[i]
    return -1


def findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound):
    '''
    Function that finds the value of m whre the r at which FNN ratio hits zero vs m curve flattens(defined by bound value)

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
         amount of delay

    m    : int
         number of embedding dimensions

    r    : double
         ratio parameter

    sig : double
         standard deviation of the data

    delta: double
         the tolerance value(the maximum difference from zero) for a value of FNN ratio to be effectively considered to be zero

    Rmin : double
         minimum value of r from where we would start the parameter search

    Rmax : double
         maximum value of r for defining the upper limit of parameter search

    rdiv : Int
         number of divisions between Rmin and Rmax for parameter search

    bound: double
         bound value for terminating the parameter serch

    Returns
    -------

    m : double
       value of embedding dimension

    References
    ----------
    - Kantz, H., & Schreiber, T. (2004). Nonlinear time series analysis (Vol. 7). Cambridge university press. section 3.3.1

    '''

    mmax = int((3 * d + 11) / 2)
    rm = fnnhitszero(u, n, d, mmax, tau, sd, delta, Rmin, Rmax, rdiv)
    rmp = fnnhitszero(u, n, d, mmax + 1, tau, sd, delta, Rmin, Rmax, rdiv)

    if (rm - rmp > bound):
        return mmax + 1
    for m in range(1, mmax):
        rmp = rm
        rm = fnnhitszero(u, n, d, mmax - m, tau, sd, delta, Rmin, Rmax, rdiv)
        print('rm-rmp:', rm - rmp)
        if (rm - rmp > bound):
            return mmax + 1 - m

    return -1


### Calculation of epsilon ###############################################

def reccplot(u, n, d, m, tau, eps):
    '''
    Function that computes the recurrence plot

    Parameters
    ----------
    u   : ndarray
        double array of shape (n,d).  Think of it as n points in a d dimensional space

    n   : int
        number of samples or observations in the time series

    d   : int
        number of measurements or dimensions of the data

    tau : int
         amount of delay

    m    : int
         number of embedding dimensions

    eps  : double
         radius of the neighbourhood for computing recurrence


    Returns
    -------

    rplot : ndarray
       recurrence plot

    References
    ----------
    - Eckmann, J.-P., Kamphorst, S. O., Ruelle, D., et al. (1995). Recurrence plots of dynamical systems. World Scientific Series on Nonlinear Science Series A, 16, 441–446.

    '''
    # normarr=[]
    s = delayseries(u, n, d, tau, m)
    rplot = np.zeros((n - (m - 1) * tau, n - (m - 1) * tau), dtype=int)
    for i in range(n - (m - 1) * tau):
        for j in range(n - (m - 1) * tau):
            # normarr.append(np.linalg.norm(s[i]-s[j]))
            if np.linalg.norm(s[i] - s[j]) < eps:
                rplot[i, j] = 1
    return rplot


def embedded_signal(
        data=None,
        rdiv=451,
        Rmin=1,
        Rmax=10,
        delta=0.001,
        bound=0.2,
        reqrr=0.1,
        rr_delta=0.005,
        epsmin=0,
        epsmax=10,
        epsdiv=1001):
    (M, N) = data.shape
    data = (data - np.mean(data, axis=0, keepdims=True)) / \
        np.std(data, axis=0, keepdims=True)
    n = M
    d = N
    u = data
    sd = 3 * np.std(u)
    tau = findtau(u, n, d, 0)
    m = findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound)
    embedded = delayseries(u, n, d, tau, m)
    return embedded, tau, m


def reccrate(rplot, n):
    '''
    Function that computes the recurrence plot

    Parameters
    ----------
    rplot: ndarray
        recurrence plot

    n   : int
        length of RP


    Returns
    -------

    reccrate : double
       recurrence rate

    References
    ----------
    - Eckmann, J.-P., Kamphorst, S. O., Ruelle, D., et al. (1995). Recurrence plots of dynamical systems. World Scientific Series on Nonlinear Science Series A, 16, 441–446.

    '''
    return float(np.sum(rplot)) / (n * n)


def findeps(u, n, d, m, tau, reqrr, rr_delta, epsmin, epsmax, epsdiv):
    '''
    Function that computes the recurrence plot

    Parameters
    ----------
    u   : ndarray
        multidimensional time series data

    n   : int
        number of observations

    d   : int
        number of dimensions

    tau : int
        amount of delay

    m   : int
        embedding dimension

    reqrr : doubld
        required recurrence rate specified in the input

    rr_delta: double
        tolerance value for considering a value of recurrence rate to be same as the one that is specified in reqrr

    epsmin : double
        lower bound for the parameter search for epsilon(neighbourhood radius)

    epsmax : double
        upper bound for the parameter search for epsilon(neighbourhood radius)

    epsdiv : double
        number of divisions for the parameter search for epsilon(neighbourhood radius) between epsmin and epsmax


    Returns
    -------

    eps   : double
       epsilon(neighbourhood radius)

    References
    ----------
    - Eckmann, J.-P., Kamphorst, S. O., Ruelle, D., et al. (1995). Recurrence plots of dynamical systems. World Scientific Series on Nonlinear Science Series A, 16, 441–446.

    '''
    eps = np.linspace(epsmin, epsmax, epsdiv)
    s = delayseries(u, n, d, tau, m)
    for k in range(epsdiv):
        rplot = np.zeros((n - (m - 1) * tau, n - (m - 1) * tau), dtype=int)
        for i in range(n - (m - 1) * tau):
            for j in range(n - (m - 1) * tau):
                if np.linalg.norm(s[i] - s[j]) < eps[k]:
                    rplot[i, j] = 1
        rr = reccrate(rplot, n - (m - 1) * tau)
        if np.abs(rr - reqrr) < rr_delta:
            return eps[k]

    return -1

### Calculation of RQA parameters ########################################


def plotwindow(M, n, win, i, j):
    window = np.zeros((win, win))
    for a in range(win):
        for b in range(win):
            window[a, b] = M[i + a, j + b]
    return window


def vert_hist(M, n):  # Functio to calculate vertical line distribution
    '''
    Function to compute vertical line distribution(counts of line lengths)

    Parameters
    ----------
    m   : ndarray
        recurrence plot

    n   : int
        length of RP

    Returns
    -------

    nvert : array
       an array containing counts of line lengths

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    nvert = np.zeros(n + 1)
    for i in range(n):
        counter = 0
        for j in range(n):
            if M[j][i] == 1:
                counter += 1
            else:
                nvert[int(counter)] += 1
                counter = 0
        nvert[counter] += 1
    return nvert


def onedhist(M, n):
    hst = np.zeros(n + 1)
    counter = 0
    for i in range(n):
        if M[i] == 1:
            counter += 1
        else:
            hst[counter] += 1
            counter = 0
    hst[counter] += 1
    return hst


def diaghist(M, n):  # Function to calculate diagonal line distribution
    '''
    Function to compute diagonal line distribution(counts of line lengths)

    Parameters
    ----------

    m   : ndarray
        recurrence plot

    n   : int
        length of RP

    Returns
    -------

    nvert : array
       an array containing counts of line lengths

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    dghist = np.zeros(n + 1)
    for i in range(n):
        diag = np.zeros(n - i)
        for j in range(n - i):
            diag[j] = M[i + j][j]
        subdiaghist = onedhist(diag, n - i)
        for k in range(n - i + 1):
            dghist[k] += subdiaghist[k]
    dghist *= 2
    dghist[n] /= 2
    return dghist


### Measures to capture probability distributions ########################

def percentmorethan(hst, mini, n):
    '''
    Function to compute determinism and laminarity from the histogram distribution of lines

    The idea is to see what fraction of rcurrent points are part of a linear structure, which can be either
    vertical or horizontal. The definition of a linear structure on an RP is based on the minimum length(mini)
    given as an input

    Parameters
    ----------
    hst   : array
        histogram counts of line lengths

    mini  : int
        minimum length of consecutive occurances of value 1 in the RP(either vertically or horizontally) that is considered as a line

    n   : int
        length of RP

    Returns
    -------

    nvert : array
       an array containing counts of line lengths

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    numer = 0
    denom = 10**(-7)
    for i in range(mini, n + 1):
        numer += i * hst[i]
    for i in range(1, n + 1):
        denom += i * hst[i]
    return numer / denom


def mode(hst, mini, n):
    '''
    Function to find mode of the line distributions

    Parameters
    ----------
    hst   : array
        histogram counts of line lengths

    mini  : int
        minimum length of consecutive occurances of value 1 in the RP(either vertically or horizontally) that is considered as a line

    n   : int
        length of RP

    Returns
    -------

    p : int
       mode of line length distribution

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    p = mini
    for i in range(mini + 1, n + 1):
        if hst[i] > hst[p]:
            p = i
    return p


def maxi(hst, mini, n):
    '''
    maximum value in the line length distribution

    Parameters
    ----------
    hst   : array
        histogram counts of line lengths

    mini  : int
        minimum length of consecutive occurances of value 1 in the RP(either vertically or horizontally) that is considered as a line

    n   : int
        length of RP

    Returns
    -------

    plmax : int
       max of line length distribution

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    lmax = 1
    for i in range(1, n - 1):
        if hst[i] != 0:
            lmax = i
    return lmax


def average(hst, mini, n):
    '''
    Function to find mean of the line distributions

    Parameters
    ----------
    hst   : array
        histogram counts of line lengths

    mini  : int
        minimum length of consecutive occurances of value 1 in the RP(either vertically or horizontally) that is considered as a line

    n   : int
        length of RP

    Returns
    -------

    mu : double
       mean of line length distribution

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    numer = 0
    denom = 10**(-7)
    for i in range(mini, n + 1):
        numer += i * hst[i]
        denom += hst[i]
    return numer / denom


def entropy(hst, mini, n):
    '''
    Function to find entropy of the line distributions

    Parameters
    ----------
    hst   : array
        histogram counts of line lengths

    mini  : int
        minimum length of consecutive occurances of value 1 in the RP(either vertically or horizontally) that is considered as a line

    n   : int
        length of RP

    Returns
    -------

    ent : int
       entropy of line length distribution

    References
    ----------
    - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
    - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
    - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
    - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

    '''
    summ = 0
    entr = 0
    for i in range(mini, n + 1):
        summ += hst[i]
    for i in range(mini, n + 1):
        if (hst[i] != 0):
            entr -= (hst[i] / summ) * np.log(hst[i] / summ)
    return entr
