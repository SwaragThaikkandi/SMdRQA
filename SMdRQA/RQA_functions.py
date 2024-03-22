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
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import csv
from tqdm import tqdm
import os
import numpy as np
import operator
import contextlib
import functools
import operator
import warnings
from numpy.core import overrides
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import csv
from tqdm import tqdm
import pickle
import random
from scipy.stats import skew
from p_tqdm import p_map
from functools import partial
from scipy.interpolate import pchip_interpolate
import memory_profiler

def doanes_formula(data,n):
    '''
    Functions for computing number of bins for data using Doane's formula
    data : input array
    n : number of elements in data
    '''
    
    g1 = skew(data)
    sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))

    k = 1 + np.log2(n) + np.log2(1 + np.abs(g1) / sigma_g1)
    
    for i in range(len(k)):
      k[i] = int(np.ceil((2/3)*k[i]))  # Round up to the nearest integer

    return k
    
def binscalc(X,n,d, method):
    """
    Function to calculate number of bins in a histogram using a generalised Freedman–Diaconis rule.
    
    Arguments:  X    double array of shape (n,d)    time series. Think of it as n points in a d dimensional space
                n    int                            number of samples in time series
                d    int                            number of measurements at each time step
                
    Output:          int array of shape (d)         Number of bins along each axis
    """
    if method == 'FD': #generalised Freedman–Diaconis rule.
      mult_fact=(n**(1/(2+d+10**(-9)))) # Generalised Freedman-Diaconis rule with Bins[i] \propto n^(1/(d+2))
      Bins=np.zeros(d)
      inf_arr=np.amin(X,axis=0)
      sup_arr=np.amax(X,axis=0)
      q25=np.quantile(X,0.25,axis=0)
      q75=np.quantile(X,0.75,axis=0)
      IQR=(q75-q25)+10**(-9)
      print('IQR:',IQR)
      Bins_1=np.ceil(mult_fact*(sup_arr-inf_arr)/(2*IQR))
      Bins=Bins_1.astype(int)
    elif method== 'sqrt':
      ns=[n]*d
      Bins_1=np.ceil(np.sqrt(ns))
      Bins=Bins_1.astype(int)
      
    elif method == 'rice':
      ns=[n]*d
      Bins_1=2*np.ceil(np.cbrt(ns))
      Bins=Bins_1.astype(int)
    elif method == 'sturges':
      ns=[n]*d
      Bins_1=1+np.ceil(np.log(ns))
      Bins=Bins_1.astype(int)
    elif method== 'doanes':
      Bins_1=doanes_formula(X,n)
      Bins=Bins_1.astype(int)
      
    elif method == 'default':
      Bins=15
      
    
    #print('inf:',inf)
    #print('sup:',sup)
    #print('IQR:',iqr)
     
    return Bins
    

def mutualinfo(X,Y,n,d):
    """
    Function to calculate mutual information between to time series
    
    Arguments:  X,Y  double array of shape (n,d)    time serieses
                n    int                            number of samples in time series
                d    int                            number of measurements at each time step
                
    Output:          double                         mutual information between X,Y
    """
    points=np.concatenate((X,Y),axis=1)
    bins=binscalc(points,n,2*d,'FD')   
    print('BINS:', bins)    
    ###print('bins:',bins.shape)
    ###print('points:',points.shape)
    p_xy=np.histogramdd(points,bins=binscalc(points,n,2*d,'default'))[0]+10**(-9) # 10^-9 added so that x log x does not diverge when x=0 in the calculation of mutual information
    p_x=np.histogramdd(X,bins=binscalc(X,n,d,'default'))[0]+10**(-9)
    p_y=np.histogramdd(Y,bins=binscalc(Y,n,d,'default'))[0]+10**(-9)
    p_xy/=np.sum(p_xy) # Normalising the probability distribution
    p_x/=np.sum(p_x)
    p_y/=np.sum(p_y)
    return np.sum(p_xy*np.log2(p_xy))-np.sum(p_x*np.log2(p_x))-np.sum(p_y*np.log2(p_y)) # formula for mutual information

def timedelayMI(u,n,d,tau):
    """
    Function to calculate mutual information between a time series and a delayed version of itself
    
    Arguments:  u    double array of shape (n,d)    time series
                n    int                            number of samples in time series
                d    int                            number of measurements at each time step
                tau  double                         amount of delay
                
    Output:          double                         mutual information between u and u delayed by tau
    """
    X=u[0:n-tau,:]
    Y=u[tau:n,:]
    return mutualinfo(X,Y,n-tau,d)
    
def findtau(u,n,d,grp):
    """
    Function to calculate correct delay for estimating embedding dimension
    
    Arguments:  u    double array of shape (n,d)    time series
                n    int                            number of samples in time series
                d    int                            number of measurements at each time step
                
    Output:          double                         delay such that the mutual information between u and u delayed by tau is at its first minimum as a function of tau
    """
    TAU=[]
    MIARR=[]
    minMI=timedelayMI(u,n,d,1)
    for tau in range(2,n):
        nextMI=timedelayMI(u,n,d,tau)
        TAU.append(tau)
        MIARR.append(nextMI)
        if nextMI>minMI:
          break
        minMI=nextMI
        
        
    
    return tau-1

#### Calculation of m ###################################################################################################################################################################

def delayseries(u,n,d,tau,m):
    """Delay series"""
    s=np.zeros((n-(m-1)*tau,m,d))
    for i in range(n-(m-1)*tau):
        for j in range(m):
            s[i,j]=u[i+j*tau]
    return s

def nearest(s,n,d,tau,m):
    nn=np.zeros(n-m*tau,dtype=int)
    nn[0]=n-m*tau-1
    for i in range(n-m*tau):
        for j in range(n-m*tau):
            if(i!=j and np.linalg.norm(s[i]-s[j])<np.linalg.norm(s[i]-s[nn[i]])):
                nn[i]=j
    return nn
                
def fnnratio(u,n,d,m,tau,r,sig):
    """Calculates the ratio of false nearest neighbours. Vary m to find stabilising trend"""
    s1=delayseries(u,n,d,tau,m)     # embedding in m dimensions
    s2=delayseries(u,n,d,tau,m+1)   # embedding in m+1 dimensions
    nn=nearest(s1,n,d,tau,m)        # containg nearest neghbours after embedding in m dimensions
    isneigh=np.zeros(n-m*tau)
    isfalse=np.zeros(n-m*tau)
    for i in range(n-m*tau):
        disto=np.linalg.norm(s1[i]-s1[nn[i]])+10**(-9)
        distp=np.linalg.norm(s2[i]-s2[nn[i]])
        if(disto<sig/r):
            isneigh[i]=1
            if(distp/disto>r):
                isfalse[i]=1
    return sum(isneigh*isfalse)/(sum(isneigh)+10**(-9))

def fnnhitszero(u,n,d,m,tau,sig,delta,Rmin,Rmax,rdiv):
    Rarr=np.linspace(Rmin,Rmax,rdiv)
    for i in range(rdiv):
        if fnnratio(u,n,d,m,tau,Rarr[i],sig)<delta:
            return Rarr[i]
    return -1
    
def findm(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound):
   
    mmax=int((3*d+11)/2)
    rm=fnnhitszero(u,n,d,mmax,tau,sd,delta,Rmin,Rmax,rdiv)
    rmp=fnnhitszero(u,n,d,mmax+1,tau,sd,delta,Rmin,Rmax,rdiv)
    
    if(rm-rmp>bound):
        return mmax+1
    for m in range(1,mmax):
        rmp=rm
        rm=fnnhitszero(u,n,d,mmax-m,tau,sd,delta,Rmin,Rmax,rdiv)
        print('rm-rmp:',rm-rmp)
        if(rm-rmp>bound):
            return mmax+1-m
              
          
            
    return -1 
      
    
    
### Calculation of epsilon ######################################################################################################################################################
   
def reccplot(u,n,d,m,tau,eps):
    """"Creates reccurrence plot"""
    #normarr=[]
    s=delayseries(u,n,d,tau,m) 
    rplot=np.zeros((n-(m-1)*tau,n-(m-1)*tau),dtype=int)
    for i in range(n-(m-1)*tau):
        for j in range(n-(m-1)*tau):
            #normarr.append(np.linalg.norm(s[i]-s[j]))
            if np.linalg.norm(s[i]-s[j])<eps:
                rplot[i,j]=1          
    return rplot
    
def reccrate(rplot,n):
    return float(np.sum(rplot))/(n*n)    
    
def findeps(u,n,d,m,tau,reqrr,rr_delta,epsmin,epsmax,epsdiv):
    eps=np.linspace(epsmin,epsmax,epsdiv)
    s=delayseries(u,n,d,tau,m) 
    for k in range(epsdiv):
        rplot=np.zeros((n-(m-1)*tau,n-(m-1)*tau),dtype=int)
        for i in range(n-(m-1)*tau):
            for j in range(n-(m-1)*tau):
                if np.linalg.norm(s[i]-s[j])<eps[k]:
                    rplot[i,j]=1
        rr=reccrate(rplot,n-(m-1)*tau)
        if np.abs(rr-reqrr)<rr_delta:
            return eps[k]
        
          
    return -1

### Calculation of RQA parameters #######################################################################################################################################################

def plotwindow(M,n,win,i,j):
    window=np.zeros((win,win))
    for a in range(win):
        for b in range(win):
            window[a,b]=M[i+a,j+b]
    return window

def vert_hist(M,n):   ##### Functio to calculate vertical line distribution
    nvert=np.zeros(n+1)
    for i in range(n):
        counter=0
        for j in range(n):
            if M[j][i]==1:
                counter+=1
            else:
                nvert[int(counter)]+=1
                counter=0
        nvert[counter]+=1
    return nvert

def onedhist(M,n):
    hst=np.zeros(n+1)
    counter=0
    for i in range(n):
        if M[i]==1:
            counter+=1
        else:
            hst[counter]+=1
            counter=0
    hst[counter]+=1
    return hst

def diaghist(M,n):   #Function to calculate diagonal line distribution
    dghist=np.zeros(n+1)
    for i in range(n):
        diag=np.zeros(n-i)
        for j in range(n-i):
            diag[j]=M[i+j][j]
        subdiaghist=onedhist(diag,n-i)
        for k in range(n-i+1):
            dghist[k]+=subdiaghist[k]
    dghist*=2
    dghist[n]/=2
    return dghist


    
### Measures to capture probability distributions #######################################################################################################################################

def percentmorethan(hst,mini,n):
    numer=0
    denom=10**(-7)
    for i in range(mini,n+1):
        numer+=i*hst[i]
    for i in range(1,n+1):
        denom+=i*hst[i]
    return numer/denom

def mode(hst,mini,n):
    p=mini
    for i in range(mini+1,n+1):
        if hst[i]>hst[p]:
            p=i
    return p
    
def maxi(hst,mini,n):
    lmax=1
    for i in range(1,n-1):
      if hst[i]!=0:
        lmax=i
    return lmax

def average(hst,mini,n):
    numer=0
    denom=10**(-7)
    for i in range(mini,n+1):
        numer+=i*hst[i]
        denom+=hst[i]
    return numer/denom
    
def entropy(hst,mini,n):
    summ=0
    entr=0
    for i in range(mini,n+1):
        summ+=hst[i]
    for i in range(mini,n+1):
        if(hst[i]!=0):
            entr-=(hst[i]/summ)*np.log(hst[i]/summ)
    return entr
    

