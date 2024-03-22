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
import ast
from SMdRQA.RQA_functions import *

    
def test_findtau():
    SIZE = 10
    np.random.seed(seed=301)
    angle = np.linspace(0, 64*np.pi, 9600)
    n = int(((1200)/4)*(12-SIZE))
    d = 1
    angle=angle[0:n]
    u=np.zeros((n,d))
    var = np.sin((2*np.pi*np.random.uniform(0,1))+(4*angle))
    u[:,0] = (var - np.mean(var))/np.std(var)
    sd=3*np.std(u)
    tau=findtau(u,n,d,0)
    assert ((tau > 0) and (tau < n))
    
def test_findm():
    SIZE = 10
    rdiv=451
    Rmin=1
    Rmax=10

    delta=0.001
    bound=0.2
    np.random.seed(seed=301)
    angle = np.linspace(0, 64*np.pi, 9600)
    n = int(((1200)/4)*(12-SIZE))
    d = 1
    angle=angle[0:n]
    u=np.zeros((n,d))
    var = np.sin((2*np.pi*np.random.uniform(0,1))+(4*angle))
    u[:,0] = (var - np.mean(var))/np.std(var)
    sd=3*np.std(u)
    tau=findtau(u,n,d,0)
    m=findm(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound)
    assert m > 0
    
def test_findeps():
    SIZE = 10
    rdiv=451
    Rmin=1
    Rmax=10

    delta=0.001
    bound=0.2
    reqrr=0.1
    rr_delta=0.005
    epsmin=0
    epsmax=10
    epsdiv=1001
    windnumb=1
    np.random.seed(seed=301)
    angle = np.linspace(0, 64*np.pi, 9600)
    n = int(((1200)/4)*(12-SIZE))
    d = 1
    angle=angle[0:n]
    u=np.zeros((n,d))
    var = np.sin((2*np.pi*np.random.uniform(0,1))+(4*angle))
    u[:,0] = (var - np.mean(var))/np.std(var)
    sd=3*np.std(u)
    tau=findtau(u,n,d,0)
    m=findm(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound)
    eps = findeps(u,n,d,m,tau,reqrr,rr_delta,epsmin,epsmax,epsdiv)
    assert eps > 0
    

def test_recc_plot():
    SIZE = 10
    rdiv=451
    Rmin=1
    Rmax=10

    delta=0.001
    bound=0.2
    reqrr=0.1
    rr_delta=0.005
    epsmin=0
    epsmax=10
    epsdiv=1001
    windnumb=1
    np.random.seed(seed=301)
    angle = np.linspace(0, 64*np.pi, 9600)
    n = int(((1200)/4)*(12-SIZE))
    d = 1
    angle=angle[0:n]
    u=np.zeros((n,d))
    var = np.sin((2*np.pi*np.random.uniform(0,1))+(4*angle))
    u[:,0] = (var - np.mean(var))/np.std(var)
    sd=3*np.std(u)
    tau=findtau(u,n,d,0)
    m=findm(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound)
    eps = findeps(u,n,d,m,tau,reqrr,rr_delta,epsmin,epsmax,epsdiv)
    rplot=reccplot(u,n,d,m,tau,eps)
    (M,N) = rplot.shape
    
    assert M == N
    assert M > 0
    
