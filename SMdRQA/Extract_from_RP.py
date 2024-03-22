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
import scipy.stats as ss

def Mode(array):
  '''
  Function to compute mode from a continuous data
  '''
  (counts, edges)=np.histogram(array)
  avgs=(edges[1:]+edges[:-1])/2
  avg=avgs[counts==np.amax(counts,keepdims=True)[0]]
  return avg[0]
  
def Check_Int_Array(array):
  '''
  This function checks whether an array consists of integers or not
  '''
  N=len(array)
  sum=0
  for i in range(N):
    if array[i]==int(array[i]):
       sum=sum+1

  if N==sum:
    out=1
  elif N!=sum:
    out=0

  return out
  
def Sliding_window(RP, maxsize, winsize):
  '''
  This is a function that would extract RQA variables from each sliding window for a given RP and for a specified window size
  Inputs _______________________________________________________________________
  RP                            : recurrence plot(numpy 2D matrix)
  maxsize                       : maximum size of the recurrence plot
  winsize                       : size of the window we are using
  Outputs_______________________________________________________________________
  Dict                          : dictionary containing RQA variables from all sliding windows, keys are window number
  '''
  num_windows=maxsize-winsize+1
  Dict={}
  for i in range(num_windows):
    sub_dict={}
    sub_window=RP[i:i+winsize,i:i+winsize]
    sub_dict['recc_rate']=reccrate(sub_window,winsize)
    vert_his=vert_hist(sub_window,winsize)
    diag_his=diaghist(sub_window,winsize)
    sub_dict['percent_det']=percentmorethan(diag_his,2,winsize)
    sub_dict['avg_diag']=average(diag_his,2,winsize)
    sub_dict['max_diag']=maxi(diag_his,2,winsize)
    sub_dict['percent_lam']=percentmorethan(vert_his,2,winsize)
    sub_dict['avg_vert']=average(vert_his,2,winsize)
    sub_dict['vert_ent']=entropy(vert_his,2,winsize)
    sub_dict['diag_ent']=entropy(diag_his,2,winsize)
    sub_dict['vert_max']=maxi(vert_his,2,winsize)
    
    Dict[i]=sub_dict

  return Dict
  
def Whole_window(RP,maxsize):
  '''
  Inputs _______________________________________________________________________
  Function for computing the RQA variables from the whole RP
  RP                             : recurrence plot(numpy 2D matrix)
  maxsize                        : size of RP
  Outputs ______________________________________________________________________
  sub_dict                       : dictionary containing values of each of the RQA variables from the entire RP
  '''
  Dict={}
  
  sub_dict={}
  sub_dict['recc_rate']=reccrate(RP,maxsize)
  vert_his=vert_hist(RP,maxsize)
  diag_his=diaghist(RP,maxsize)
  sub_dict['percent_det']=percentmorethan(diag_his,2,maxsize)
  sub_dict['avg_diag']=average(diag_his,2,maxsize)
  sub_dict['max_diag']=maxi(diag_his,2,maxsize)
  sub_dict['percent_lam']=percentmorethan(vert_his,2,maxsize)
  sub_dict['avg_vert']=average(vert_his,2,maxsize)
  sub_dict['vert_ent']=entropy(vert_his,2,maxsize)
  sub_dict['diag_ent']=entropy(diag_his,2,maxsize)
  sub_dict['vert_max']=maxi(vert_his,2,maxsize)
    


  return sub_dict
  
def windowed_RP(winsize,RP_dir, save_path):
  '''
  Inputs _______________________________________________________________________
  This function computes RQA variable from each window and stores that in a dictionary
  type                            : Type of data, whether its GD/pre GD/randomized
  winsize                         : Size of window to be used
  RP_dir                          : directory where the RPs are stored as numpy files
  save_path                       : directory in which the dictionary should be saved in the form of a pickle file
  Output _______________________________________________________________________
  Dict                            : Dictionary containing RQA variables from the sliding windows
  '''
  dir=RP_dir
  files=os.listdir(dir)
  size_arr=[]
  Dict={}
  for file in files:
    a=file.strip().split('.')[0]  # To have file name as an identifier of the variable. Note that the file name should not contain "." other than the one before "npy"
    
    RP_s=np.load(dir+'/'+file,allow_pickle=True)
    (M,N)=RP_s.shape
    sub_Dict1=Sliding_window(RP_s, M, winsize)
    sub_dict2=Whole_window(RP_s, M)
    Dict[a]={'sliding':sub_Dict1,'Whole':sub_dict2}

  with open(save_path+'/whole_.pkl', 'wb') as handle:
    pickle.dump(Dict, handle)

  return Dict
  
def First_middle_last_avg(DICT, var, win_loc):
  '''
  This is a function to compute different summary statistics from the RQA variable distribution that we would get from the windows
  Inputs _______________________________________________________________________
  DICT_pre                          : dictionary containing RQA variables from data
  win_loc: window of interest(fist/last/mean/median etc)
  Outputs ______________________________________________________________________
  data                              : pandas dataframe containing the calculated summary statistic(specified) for the specified variable across different datasets
  '''
  Grp_IDs=list(DICT.keys())
  num_plots=len(Grp_IDs)
  vars=[]
  label=[]
  out=[]
  window=[]
  GID=[]
  for i in range(len(Grp_IDs)):
    num=Grp_IDs[i]
    grp_dyn=DICT[num]['sliding']
    sub_keys=grp_dyn.keys()
    recc_rate=[]
    wind=[]
    vars_sub4=[]
    label_sub4=[]
    for elem in sub_keys:
      

      vars_sub4.append(grp_dyn[elem][var])
      label_sub4.append(elem)

    if win_loc=='first':
      index=0
      app_var=vars_sub4[index]
    elif win_loc=='middle':
      index=int(np.ceil(len(vars_sub4)/2))
      app_var=vars_sub4[index]
      print('Length of array:',len(vars_sub4))
      print('Index of middle element:',index)
    elif win_loc=='last':
      index=-1
      app_var=vars_sub4[index]
    elif win_loc=='max':
      app_var=np.max(vars_sub4)
    elif win_loc=='avg':
      app_var=np.mean(vars_sub4)
    elif win_loc=='mode':
      Isint=Check_Int_Array(vars_sub4)
      if Isint==0:
        app_var=Mode(vars_sub4)
      elif Isint==1:
        (app_var,count)=ss.mode(vars_sub4)
    elif win_loc=='median':
      app_var=np.median(vars_sub4)
    elif win_loc=='whole':
      app_var=DICT[num]['Whole'][var]
      

      
    
    vars.append(app_var)
    label.append('Lorenz')
    window.append(win_loc)
    GID.append(num)

    

  DICT={'label':label,var:vars,'window':window,'group':GID}
  data=pd.DataFrame.from_dict(DICT)
  #sns.displot(data=data, x=var, kde=True,hue=['label','outcome'],kind='hist',norm_hist=True)
  return data
  
def First_middle_last_sliding_windows(DICT, var):
  '''
  Function to compute all summary statistics that we want for a given RQA variable
  Inputs _______________________________________________________________________
  DICT                          : dictionary containing RQA variables from data collected 
  var                               : specific RQA variable we are looking for
  Outputs ______________________________________________________________________
  full_data                         : pandas dataframe containing all summary statistics that we want for a given RQA variable
  '''
  win_locs=['avg','median','mode', 'whole']
  dfs=[]
  for win_loc in tqdm(win_locs):
    data=First_middle_last_avg(DICT,var, win_loc)
    dfs.append(data)

  full_data=pd.concat(dfs)
  a=full_data.columns[1]
  full_data=full_data.reset_index(drop=True)

  #sns.histplot(full_data,x=var)
  #g = sns.FacetGrid(full_data, col="window",  row="label")
  #g.map(sns.histplot,x=a,hue='outcome',norm_hist=True)
  return full_data
  
def First_middle_last_sliding_windows_all_vars(DICT, save_path):
  '''
  Function to compute all summary statistics that we want for all RQA variables
  Inputs _______________________________________________________________________
  DICT                          : dictionary containing RQA variables from data collected 
  Outputs ______________________________________________________________________
  data_out                         : pandas dataframe containing all summary statistics that we want for all RQA variable
  '''
  win_locs=['avg','median','mode', 'whole']
  vars=['recc_rate','percent_det','avg_diag','max_diag','percent_lam','avg_vert','vert_ent','diag_ent','vert_max']
  dfs=[]
  for var in tqdm(vars):
    df=First_middle_last_sliding_windows(DICT, var)
    dfs.append(df)

  data_out=pd.concat(dfs,axis=1)
  data_out=data_out.T.drop_duplicates().T
  #data_out.to_csv('/content/drive/MyDrive/MS20181013/NL_test/results/whole_diff_windows38.csv')
  #data_out.to_csv('whole_diff_windows.csv')
  data_out.to_csv(save_path)
  
  
