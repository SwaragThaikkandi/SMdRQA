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

def RP_computer(input_path, RP_dir,rdiv=451, Rmin=1, Rmax=10, delta=0.001, bound=0.2, reqrr=0.1, rr_delta=0.005, epsmin=0, epsmax=10, epsdiv=1001, windnumb=1):
  '''
  Input arguments_____________________________________________________________________________
  input_path       : folder containing the numpy files, rows> number of samples, columns> number of streams
  RP_dir           : directory in which the RPs should be stored
  rdiv             : number of divisions(resolution) for the variable r during parameter search for embedding dimension
  Rmin             : minimum value for the variable r during parameter search for embedding dimension
  Rmax             : maximum value for the variable r during parameter search for embedding dimension
  delta            : the tolerance value below which an FNN value will be considered as zero
  bound            : This is the value in the r value(at which FNN hits zero) va embedding dimension plot. The search is terminated if the value goes below this tolerance value and the value just below tolerance value is reported for embedding dimmension
  req_rr           : This is a variable that user can define. This controls the overall recurrence rate of the whole RP
  rr_delta         : this variable is used to define tolerance for accepting a value of neighbourhood radius. If the absolte differenece between the resccurence rate value to that of the desired value is less than this tolerance, the value of epsilon is accepted
  eps_min          : Minimum value of the neighbourhood radius value to being with for fixing the reccurrence rate
  eps_max          : Maximum value of the neighbourhood radius value above which the search won't progress
  eps_div          : Number of divisions between eps_min and eps_max
  windnumb         : number of windows getting extracted
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
      try:
        m=findm(u,n,d,tau,sd,delta,Rmin,Rmax,rdiv,bound)
        print('Done m calculation....')
        print('m:',m)
        print('starting eps calculation ...')
        eps=findeps(u,n,d,m,tau,reqrr,rr_delta,epsmin,epsmax,epsdiv)
        print('Done eps calculation....')
        print('EPS:',eps)
        rplot=reccplot(u,n,d,m,tau,eps)
        print('Done rplot calculation....')
        rplotwind=rplot
        np.save(RP_dir+'/'+File,rplotwind)
        
          #notFound = 0
        FILE.append(File)
        TAU.append(tau)
        Marr.append(m)
        EPS.append(eps)
        BOUND.append(bound)
        
      except ValueError:
        print('unable to compute due to value error')
        ERROROCC.append(File)
       
            
    except MemoryError:
      print("Couldn't do computation due to numpy.core._exceptions.MemoryError")
      ERROROCC.append(File)

    
  DICT={'error occurances': ERROROCC}
  df_out=pd.DataFrame.from_dict(DICT)
  df_out.to_csv('Error_Report_Sheet.csv')

  DICT2={'file':FILE,'tau':TAU,'m':Marr,'eps':EPS,'bound':BOUND}
  df_out2=pd.DataFrame.from_dict(DICT2)
  df_out2.to_csv('param_Sheet.csv')
