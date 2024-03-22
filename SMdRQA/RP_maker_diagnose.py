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

from SMdRQA.RQA_functions import doanes_formula
from SMdRQA.RQA_functions import binscalc
from SMdRQA.RQA_functions import mutualinfo
from SMdRQA.RQA_functions import timedelayMI
from SMdRQA.RQA_functions import findtau
from SMdRQA.RQA_functions import delayseries
from SMdRQA.RQA_functions import nearest
from SMdRQA.RQA_functions import fnnratio
from SMdRQA.RQA_functions import fnnhitszero
from SMdRQA.RQA_functions import findm
from SMdRQA.RQA_functions import reccplot
from SMdRQA.RQA_functions import reccrate
from SMdRQA.RQA_functions import findeps
from SMdRQA.RQA_functions import plotwindow
from SMdRQA.RQA_functions import vert_hist
from SMdRQA.RQA_functions import onedhist
from SMdRQA.RQA_functions import diaghist
from SMdRQA.RQA_functions import percentmorethan
from SMdRQA.RQA_functions import mode
from SMdRQA.RQA_functions import maxi
from SMdRQA.RQA_functions import average
from SMdRQA.RQA_functions import entropy
import seaborn as sns

def fnnhitszero_Plot(u,n,d,m,tau,sig,delta,Rmin,Rmax,rdiv):
    '''
    This function, insted of giving value of r that is giving sufficiently small FNN to be called as the value at which it hits zero. This is done as we need to find a value out of it
    returns:
    Rarr           : array of values of r values
    FNN            : corresponding false nearest neighbour values
    '''
    Rarr=np.linspace(Rmin,Rmax,rdiv)
    FNN = []
    for i in range(rdiv):
        FNN.append(fnnratio(u,n,d,m,tau,Rarr[i],sig))
            
    return Rarr, FNN
    
def findm_Plot(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound, save_path, plot=False):
    if plot == True:
      mmax=int((3*d+11)/2)
      
      DICT = {}
      plt.figure(figsize = (16,12))
      for m in range(1,mmax):
          color = plt.cm.rainbow(m/(mmax-1))
          Rarr, FNN=fnnhitszero_Plot(u,n,d,mmax-m,tau,sd,delta,Rmin,Rmax,rdiv)
          label = 'm='+str(mmax-m)
          plt.plot(Rarr , FNN, color = color, label = label)
          DICT[mmax-m] = {'r':Rarr, 'FNN':FNN}
    
      plt.xlabel('r')
      plt.ylabel('FNN')
      plt.title('r vs FNN plot')
      plt.legend()
      plt.savefig(save_path+'.pdf')
      with open(save_path+'.pickle', 'wb') as handle:
          pickle.dump(DICT, handle)

def RP_diagnose(input_path, diagnose_dir,rdiv=451, Rmin=1, Rmax=10, delta=0.001, bound=0.2):
  '''
  Input arguments_____________________________________________________________________________
  input_path       : folder containing the numpy files, rows> number of samples, columns> number of streams
  diagnose_dir     : directory in which the pickles containing FNN vs r data should be stored
  rdiv             : number of divisions(resolution) for the variable r during parameter search for embedding dimension
  Rmin             : minimum value for the variable r during parameter search for embedding dimension
  Rmax             : maximum value for the variable r during parameter search for embedding dimension
  delta            : the tolerance value below which an FNN value will be considered as zero
  bound            : This is the value in the r value(at which FNN hits zero) va embedding dimension plot. The search is terminated if the value goes below this tolerance value and the value just below tolerance value is reported for embedding dimmension
  
  Output_______________________________________________________________________________________
  
  RPs              : recurrence plots saved as npy files in the given directory
  Error_Report_Sheet : Analogous to a log file, will record those instances where RP computation failed either due to memory error or due to value error
  param_Sheet      : RQA parameter estimated for each files
  '''
  path=input_path
  files=os.listdir(path)
  ERROROCC=[]
  FILE=[]
  TAU=[]
  Marr=[]
  EPS=[]
  BOUND=[]
  
  for File in tqdm(files):
    try:
      file_path=path+'/'+File
      data=np.load(file_path)
      (M,N)=data.shape
    
      data = (data - np.mean(data, axis=0, keepdims=True))/np.std(data, axis=0, keepdims=True)
        
      
      n=M
      d=N
      u=data
      
      sd=3*np.std(u)
      print('starting tau calculation ...')
      tau=findtau(u,n,d,0)
      print('Done Tau calculation....')
      print('TAU:',tau)
      print('starting m calculation ...')
      #notFound = 1
      #while notFound == 1: 
        #try: 
      File_out = File.split('.')[0]
      try:
        findm_Plot(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound,diagnose_dir+'/'+File_out,plot=True)
        
        
      except ValueError:
        print('unable to compute due to value error')
        ERROROCC.append(File)
       
            
    except MemoryError:
      print("Couldn't do computation due to numpy.core._exceptions.MemoryError")
      ERROROCC.append(File)
      
  DICT={'error occurances': ERROROCC}
  df_out=pd.DataFrame.from_dict(DICT)
  df_out.to_csv('Error_Report_Sheet.csv')  

    
def get_minFNN_distribution_plot(path, save_name):
  '''
  This is a function used to get the delta value or the value used to effectively consider 
  a particular value of FNN effectively as zero. This function estimates and plots the distribution
  of minimum FNN value for different embedding dimensions(m). It computes the upper(2.5% quantile)
  and lower(97.5% quantile) and when we want to get r(at FNN hitting zero) vs m graph, generally the 
  delta value should be more than the highest upper bound(most likely for m=1) is choosen
  Input:__________________________________________________________________________________________
  path    : path to folder containing pickes files computed using "RP_diagnose" function
  savename: The output plot and CSV file would be saved in the name specified
  '''
  files = os.listdir(path)
  M=[]
  FILE=[]
  MIN_fnn = []

  for File in files:
    open_path = path+'/'+File
    with open(open_path, 'rb') as f:
      Dict = pickle.load(f)

    keys = list(Dict.keys())
    for key in keys:
      sub_dict = Dict[key]
      fnn = sub_dict['FNN']
      r = sub_dict['r']
      min_fnn = np.min(fnn)
      FILE.append(File)
      M.append(key)
      MIN_fnn.append(min_fnn)

  DICT = {'file':FILE,'m':M,'min_fnn':MIN_fnn}
  df_out = pd.DataFrame.from_dict(DICT)
  plt.figure(figsize = (12,9))
  sns.violinplot(data = df_out, x='m',y='min_fnn')
  plt.savefig(save_name+'.png')
  Ms = np.unique(np.array(df_out['m']))
  low_b = []
  up_b = []
  for m in Ms:
    df_out_sub = df_out[df_out['m']==m]
    mfnn = np.array(df_out_sub['min_fnn'])
    low_b.append(np.quantile(mfnn,0.025))
    up_b.append(np.quantile(mfnn,0.975))

  DICT2 = {'m':Ms, '2.5% quantile':low_b, '97.5% quantile': up_b}
  df_out2 = pd.DataFrame.from_dict(DICT2)
  df_out2.to_csv(save_name+'.csv')


