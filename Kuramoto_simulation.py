########################################################## import packahges #################################################################################################
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

from kuramoto import Kuramoto, plot_phase_coherence, plot_activity # see https://github.com/fabridamicelli/kuramoto
# For installing the kuramoto package run: pip install kuramoto

from RP_maker import RP_computer

from Extract_from_RP import Mode
from Extract_from_RP import Check_Int_Array
from Extract_from_RP import Sliding_window
from Extract_from_RP import Whole_window
from Extract_from_RP import windowed_RP
from Extract_from_RP import First_middle_last_avg
from Extract_from_RP import First_middle_last_sliding_windows
from Extract_from_RP import First_middle_last_sliding_windows_all_vars

from cross_validation import feature_selection
from cross_validation import nested_cv
import networkx as nx
############################################################### Define function for adding noise ############################################################################
def add_noise(signal, snr):
    """
    Add Gaussian noise to a signal with a specific signal-to-noise ratio (SNR).
    
    Parameters:
    signal (array): Input signal.
    snr (float): Signal-to-noise ratio (SNR) as a ratio.
    
    Returns:
    noisy_signal (array): Signal with added noise.
    """
    # Clip signal to maximum value of 1 to prevent overflow errors
    
    # Calculate signal power
    signal_power = np.mean(np.var(signal))
    
    # Calculate noise power
    noise_power = signal_power / snr
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    # Add noise to signal
    noisy_signal = signal + noise
    
    # Clip noisy signal to maximum value of 1 to prevent overflow errors
    #noisy_signal = np.clip(noisy_signal, np.min(signal), np.max(signal))
    
    return noisy_signal

########################################################################### Simulate Data ##################################################################################################

SNRu=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]                                      # These are the range of SNR values
np.random.seed(301)
random.seed(301)
for i in tqdm(range(10)):
  N=random.sample(range(3, 7), 1)[0]                                                # Selecting a random size for the number of oscillators, between 3 and 6
  omega = np.random.normal(0,1,size=N)                                              # Sampling the natural frequencies of the oscillators from a normal distribution
  Ks=(np.max(omega)-np.min(omega))                                                  # Computing the critical coupling strength for the population of oscillators
  K_sample=np.random.uniform(low=0, high=2*Ks,size=10)                              # Sampling the coupling strength from a uniform distribution from 0 to 2Ks so 
                                                                                    # that there is equal probability for finding a system having coupling strength below and 
                                                                                    # above critical coupling strength
  rp_sizes = random.sample(range(15, 45), len(K_sample))                            # Selecting a random length for the time series
  graph_nx = nx.erdos_renyi_graph(n=N, p=1)                                         # Defining an all-to-all connected graph
  graph = nx.to_numpy_array(graph_nx)                                               # Converting connection matrix to numpy matrix
  for k2 in tqdm(range(len(K_sample))):                                             # Looping over different coupling strengths
    K_used=K_sample[k2]                               
    model = Kuramoto(coupling=K_used, dt=0.01, T=rp_sizes[k2], n_nodes=N,natfreqs=omega)  # Defining the Kuramoto model, dt is the timestep
    act_mat = model.run(adj_mat=graph)                                              # Activation matrix from running the model
    THETA=act_mat.T                                                                 # Taking transpose to get data from each oscillator as a column
    for snr in tqdm(SNRu):                                                          # Loop over different signal to noise ratios
      signals=THETA
      for j in range(N):                                                            # Add noise to the signals
        signals[:,j]=add_noise(signals[:,j], snr)
        
      np.save('/user/swarag/Kuramoto/signals/('+str(snr)+','+str(N)+','+str(Ks)+','+str(K_used)+','+str(rp_sizes[k2])+')-.npy',signals)  # Save the data to a numpy file
    

###################################################################### Generate RPs ########################################################################################################
input_path = '/user/swarag/Kuramoto/signals'                                        # directory to which the signals are saved
RP_dir = '/user/swarag/Kuramoto/RP'                                                 # directory to which we want to save the RPs
RP_computer(input_path, RP_dir)                                                     # generating RPs and saving to the specified folder
###################################################################### Extracting Data From RPs ############################################################################################
Dict_RPs=windowed_RP(68, 'RP')                                                      # Specifying window size and folder in which RPs are saved
First_middle_last_sliding_windows_all_vars(Dict_RPs,'Kuramoto_data.csv')            # Saving RQA variables to a csv file
###################################################################### Process data #######################################################################################################
data = pd.read_csv('Kuramoto_data.csv')
FILE = np.array(data['group'])                                                      # In the output data, the field named 'group' will have file name which contains details
SNR =[]
NUM =[]
Kc =[]
K =[]
SIZE =[]
for FI in FILE:
  info = ast.literal_eval(FI)
  SNR.append(info[0])
  NUM.append(info[1])
  Kc.append(info[2])
  K.append(info[3])
  SIZE.append(info[4])
  
data['snr'] = SNR
data['N'] = NUM
data['Kc'] = Kc
data['K'] = K
data['length'] = SIZE
K = np.array(K)
Kc = np.array(Kc)
SYNCH = 1*(K>Kc)                                                                   # Defining synchrony condition, coupling strength grater than that of the critical coupling strength
data['synch'] = SYNCH
################################################################## Select the value of SNR ################################################################################################
data_ = data[data['snr']==1.0].reset_index(drop = True)
################################################################## Scale the data #########################################################################################################\
features=['recc_rate',
 'percent_det',
 'avg_diag',
 'max_diag',
 'percent_lam',
 'avg_vert',
 'vert_ent',
 'diag_ent',
 'vert_max']
for feature in features:
  arr = np.array(data_[feature])
  data_[feature] = (arr - np.mean(arr))/(np.std(arr) + 10**(-9))
  
################################################################# Run the classification ###################################################################################################
nested_cv(data_, features, 'synch', 'Kuramotot(SNR=1.0)', repeats=100, inner_repeats=10, outer_splits=3, inner_splits=2)
#################################################################   DONE ! ################################################################################################################
