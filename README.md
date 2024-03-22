[![Python package](https://github.com/SwaragThaikkandi/SMdRQA/actions/workflows/python-package.yml/badge.svg)](https://github.com/SwaragThaikkandi/SMdRQA/actions/workflows/python-package.yml)
[![Upload to PIP](https://github.com/SwaragThaikkandi/SMdRQA/actions/workflows/python-publish.yml/badge.svg)](https://github.com/SwaragThaikkandi/SMdRQA/actions/workflows/python-publish.yml)

# SMdRQA: Implementing Sliding Window MdRQA to get Summary Statistics Estimate of MdRQA measures from the Data

This is a step by step tutorial about how to use the functions provided in the github repository. Here, we are providing codes, using which the RQA measures can be estimated after a parameter search for finding the embedding dimension. 
## Installing SMdRQA
You can install the package using PyPI
```shell-session
pip install SMdRQA
```
## import packages 
We will begin with importing packages and functions. Note that the python files should be copied to the mail analysis directory
```python
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

from SMdRQA.RP_maker import RP_computer

from SMdRQA.Extract_from_RP import Mode
from SMdRQA.Extract_from_RP import Check_Int_Array
from SMdRQA.Extract_from_RP import Sliding_window
from SMdRQA.Extract_from_RP import Whole_window
from SMdRQA.Extract_from_RP import windowed_RP
from SMdRQA.Extract_from_RP import First_middle_last_avg
from SMdRQA.Extract_from_RP import First_middle_last_sliding_windows
from SMdRQA.Extract_from_RP import First_middle_last_sliding_windows_all_vars

from SMdRQA.cross_validation import feature_selection
from SMdRQA.cross_validation import nested_cv
```
## Example 1: Kuramoto Model
In this example, we will be simulating the Kuramoto model, varying number of scillators, length of time series, and the coupling strength. The number of oscillators will be randomly choosen from a descrete uniform distribution of integers from 3 to 6. Time series length is also choosen similarly. Coupling strength is sampled from a continuous uniform distribution from 0 to 2Kc, where Kc is the critical coupling strength. 
We are using the mean field Kuramoto model, where the coupling strength between any two oscillator is the same. The system is given by the differential equation: 

$$\dot{\theta_{i}} = \omega_{i} + \sum_{j=1}^{N} K_{ij} \sin{(\theta_{j}-\theta{i})}, i=1, ..., N$$


We will sample the frequencies($$\omega_{i}$$) from a standard normal distribution. And the critical coupling strength for mean field model is given by:


$$K_{c} = |\omega_{max}-\omega_{min}|$$


And in the case of mean field model, the coupling strength is given by: 


$$K_{ij} = K/N >0, \forall i,j \in \{1,2,3,...,N\}$$


Here, synchrony of the system is defined in terms of a complex valued order parameter($$r$$), which is given by:


$$r e^{i\psi} =  \frac{1}{N} \sum_{j=1}^{N}e^{i \theta_{j}}$$

Here $$\psi$$ is the average phase value. To arrive at  an expression that makes the dependence of synchrony on values of K explicit, we begin with multiplying both sides by $$e^{-i \theta_{i}}$$.

$$r e^{i\psi} e^{-i \theta_{i}} = \left( \frac{1}{N} \sum_{j=1}^{N}e^{i \theta_{j}}\right) e^{-i \theta_{i}}$$

$$r e^{i(\psi -\theta_{i})}= \frac{1}{N} \sum_{j=1}^{N}e^{i(\theta_{j}-\theta_{i})}$$

$$\dot{\theta_{i}} = \omega_{i} + K r \sin{(\psi -\theta_{i})}$$

Here, when the coupling strength is tending to zero, the oscillators would be oscillating in their natural frequencies. 
### What is the order parameter? How does it matter?
<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Fig14-1.png" style="width: 750px;">
  
</div>
Figure shows order parameter plotted against time, and the color coding is for the coupling strength. We can see that, above the critical couping strengh, the critical coupling strength increases and stays there. Below critical coupling strength also, we can see high r values, but, it is not sustained. So, we can see that, more that r value itself, estimated at a point in time, coupling strength is more informative about the system. Now, we had seen this we can go through the analysis pipeline:
First we define a function to add noise to the signals. This function will change the signal to noise ratio of the signals


```python


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
```

Now, we will simulate the system using Kuramoto package available at "https://github.com/fabridamicelli/kuramoto". We will simulate the system, add noise and will save it to a directory( as numpy files). These files will be later accessed for computing RPs. 


```python
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
      signals=signals_ori
      for j in range(N):                                                            # Add noise to the signals
        signals[:,j]=add_noise(signals[:,j], snr)
      signals = signals[::10,:]
      np.save('/user/swarag/Kuramoto/signals/('+str(snr)+','+str(N)+','+str(Ks)+','+str(K_used)+','+str(rp_sizes[k2])+')~.npy',signals)  # Save the data to a numpy file
    

```

Now we have saved the time series data to a folder, we can access the folder for creating RPs. Note that the folder containig the data files shouldn't contain any other files. The function we defined are making use of the "listdir" function from os package, and the code doesn't handle exceptions. Now we can generate the RPs


```python
input_path = '/user/swarag/Kuramoto/signals'                                        # directory to which the signals are saved
RP_dir = '/user/swarag/Kuramoto/RP'                                                 # directory to which we want to save the RPs
RP_computer(input_path, RP_dir)                                                     # generating RPs and saving to the specified folder
```
One may think that the function is so simple, only asking the user to input information about the directories. However, if you access the original python script, there we can see many inputs, but with some default values. Two of those default values may have to change for some datasets, about which we would discuss later in this tutorial. Now we have saved the RPs, the RPs would look like the following: 
<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Fig11C-1.png" style="width: 750px;">
  
</div>
Nest step is to compute the RQA values and their central tendency measures from the sliding windowws. An important step for that is to determine the window size, which again is subjective to the experimentor and data. Let's see how window size estimate can be done, in this case using the variable percentage determinism. 


```python
from window size import Sliding_windowBS_sub_percent_det
from window size import Sliding_windowBS
from window size import Sliding_window_whole_data

windowing_data = Sliding_window_whole_data(RP_dir, 'percent_det')

```
From this, what we will get is a dataframe, which contains the difference between 95th percentile and 5th percentile for each file. Now we have to plot it for different noise levels. 


```python
FILE = np.array(windowing_data['group'])                                                      # In the output data, the field named 'group' will have file name which contains details
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
  
windowing_data['snr'] = SNR
windowing_data['N'] = NUM
windowing_data['Kc'] = Kc
windowing_data['K'] = K
windowing_data['length'] = SIZE
```

Now since we have added these columns to the dataframe, we can use seaborn package for plotting

```python
plt.figure(figsize = (16,12))
sns.pointplot(data = windowing_data, x = 'WINSIZE', y = '95% quantile- 5% quantile', hue = 'snr')
plt.show()
```
<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/determinism_win_size.png" style="width: 750px;">
  
</div>
We considered some value near window size  = 70 to be good enough, and took 68 as the window size. Note that, due to monotonously decreasing nature of the graph, it is impossible to get an optimal value, which is not subjective. 
Now that we decided about the window size, we can estimate RQA variables from the sliding windows having the given size.


```python
Dict_RPs=windowed_RP(68, 'RP')                                                      # Specifying window size and folder in which RPs are saved
First_middle_last_sliding_windows_all_vars(Dict_RPs,'Kuramoto_data.csv')            # Saving RQA variables to a csv file
```

The resulting dataframe that is getting saved will have a column, "group", which will have the file name(excluding ".npy"). In this code, we are coding information regarding synchrony using the file name, as given below:


```python
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
```

Now, we need to select an SNR value, scale the variables and run the classifier
```python
################################################################## Select the value of SNR ################################################################################################
data_ = data[data['snr']==1.0].reset_index(drop = True)
data_ = data_[data['window']== 'mode'].reset_index(drop = True)                   # mode of RQA variables from the sliding windows
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
```

# Troubleshooting: Parameter Search for embedding dimension

If you were suspecious about the oversimplified look of the function "RP_computer", you are correct. This function also implements a parameter search method for finding the embedding dimensions under the hood. Even though some default values are there, they may not be applicable to all datasets. In such cases, we need to see, what can be set. We had to do this while analyzing data from Koul et al(2023), and it is for this, we have some functions in "RP_maker_diagnose.py". We will see how we can use these functions


```python
from RP_maker_diagnose import fnnhitszero_Plot
from RP_maker_diagnose import findm_Plot
from RP_maker_diagnose import RP_diagnose
from RP_maker_diagnose import get_minFNN_distribution_plot

input_path = '/user/swarag/Koul et al/data_npy'                                        # directory to which the signals are saved
diagnose_dir = '/user/swarag/Koul et al/diagnose'                                            # directory in which pickle files from this function are saved
RP_diagnose(input_path, diagnose_dir)
```
Now, we have saved the picke files to a directory, we can use those pickle files in the next function. Following function estimates the lower and upper bound (2.5th and 975th percentile) of the distribution of minimum false nearest beighbour(FNN) values for different embedding dimensions. It gives a plot and a CSV file. 


```python
get_minFNN_distribution_plot(path, 'Koul_et_al_RP_diagnose')
```

<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Koul_et_al_RP_diagnose.png" style="width: 750px;">
  
</div>
<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Koul_et_al_RP_diagnos_csv.png" style="width: 300px;">
  
</div>
We can see that the minimum value of FNN goes down as we increases the embedding diension. But, the value of delta, which is a parameter that determines whether a particular value of embedding dimnesion can be effectively considered as zero or not, should include most of these values. Generally m=1 is less useful, hence, we can consider the upper bound from m=2 onwards. Then we can set this limit for defining the r value at which FNN hits zero and then we can find out desired value of embedding dimension by setting a threshold value for r(at which FNN hits zero) vs embedding dimension plot.


## Example 2: Rossler Attractor
Firstly, we will define functions for simulating Rossler attractor


```python
def rossler(t, X, a, b, c):
    """The Rössler equations."""
    x, y, z = X
    xp = -y - z
    yp = x + a*y
    zp = b + z*(x - c)
    return xp, yp, zp
    
def rossler_time_series(tmax,n, Xi, a, b, c):
    x, y, z = Xi
    X=[x]
    Y=[y]
    Z=[z]
    dt=0.0001
    for i in range(1, tmax):
      #print('Xi:', Xi)
      x, y, z = Xi
      xp, yp, zp=rossler(t, Xi, a, b, c)
      x_next=x+dt*xp
      y_next=y+dt*yp
      z_next=z+dt*zp
      X.append(x_next)
      Y.append(y_next)
      Z.append(z_next)
      Xi=(x+dt*xp,y+dt*yp,z+dt*zp)
      
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    
    step=int(tmax/n)
    indices = np.arange(0,tmax, step)
    #print('IS NAN X:',np.sum(np.isnan(X[indices])))
    #print('IS NAN Y:',np.sum(np.isnan(Y[indices])))
    #print('IS NAN Z:',np.sum(np.isnan(Z[indices])))
    
    return X[indices], Y[indices], Z[indices]
```
Now, define the parameter values. We are mainly varying a, while keeping b and c the same for getting transition from periodic to chaotic attractor. 


```python
b = 0.2
c = 5.7
a_array = [0.1,0.15,0.2,0.25,0.3]
SNR = [0.125,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
```

Now we will simulate the data for different nose levels, and will save it to a directory


```python
for snr in tqdm(SNR):                                                                             # Looping over SNR values
  for j in tqdm(range(len(a_array)):                                                              # Looping over values of parameter a
    a=a_array[j]
    random.seed(1)
    np.random.seed(seed=301)
    rp_sizes = random.sample(range(150, 251), 10)                                                 # selecting a length for time series
    for k in tqdm(range(5)):                                                                      # Repeats
       
          u0, v0, w0 = 0+(1*np.random.randn()), 0+(1*np.random.randn()), 10+(1*np.random.randn()) # Setting initial conditions
          
          
          for k2 in tqdm(range(len(rp_sizes))):                                                   # Looping over different RP sizes(time series length)
            tmax, n = int(1000000*(rp_sizes[k2]/250)), rp_sizes[k2]
            print('started model')
            Xi=(u0, v0, w0)
            t = np.linspace(0, tmax, n)
            x, y, z=rossler_time_series(tmax,n, Xi, a, b, c)
            
            x=add_noise(x, snr)                                                                  # Adding noise
            y=add_noise(y, snr)
            z=add_noise(z, snr)
            u[:,0]=x                                                                             # Defining the output matrix
            u[:,1]=y
            u[:,2]=z
            np.save('/user/swarag/Rossler/signals/('+str(snr)+','+str(a)+','+str(u0)+','+str(v0)+','+str(w0)+','+str(rp_sizes[k2])+')~.npy',u)  # Save the data to a numpy file
```


Since we have saved the time series data to a folder, we can repeat the same steps


```python
input_path = '/user/swarag/Rossler/signals'                                        # directory to which the signals are saved
RP_dir = '/user/swarag/Rossler/RP'                                                 # directory to which we want to save the RPs
RP_computer(input_path, RP_dir)                                                    # generating RPs and saving to the specified folder
``

Since we have generated the RPs at this step, we can extract the variables from it. We have selected same window size for the sliding window approach. 


```python
Dict_RPs=windowed_RP(68, 'RP')                                                      # Specifying window size and folder in which RPs are saved
First_middle_last_sliding_windows_all_vars(Dict_RPs,'Rossler_data.csv')            # Saving RQA variables to a csv file
```

Now we need to add additional columns to the data for later analysis


```python
data = pd.read_csv('Rossler_data.csv')
FILE = np.array(data['group'])                                                      # In the output data, the field named 'group' will have file name which contains details
SNR =[]
A =[]
U0 =[]
V0 =[]
W0 =[]
SIZE =[]
for FI in FILE:
  info = ast.literal_eval(FI)
  SNR.append(info[0])
  A.append(info[1])
  U0.append(info[2])
  V0.append(info[3])
  W0.append(info[4])
  SIZE.append(info[5])
  
data['snr'] = SNR
data['a'] = A
data['u0'] = U0
data['v0'] = V0
data['w0'] = W0
data['length'] = SIZE
A = np.array(A)

SYNCH = 1*(A>0.2)                                                                   # Defining synchrony condition, parameter value belonging to the chaotic region
data['synch'] = SYNCH
```
Now, we need to select an SNR value, scale the variables and run the classifier
```python
################################################################## Select the value of SNR ################################################################################################
data_ = data[data['snr']==1.0].reset_index(drop = True)
data_ = data_[data['window']== 'mode'].reset_index(drop = True)                   # mode of RQA variables from the sliding windows
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
```

## Example 3: Data from Koul et al(2023)
Data was originally provided in the form of CSV files, which we converted into numpy(.npy) files and that is used for later analysis. We will also show the step in which we are converting the CSV files into numpy files, as at this stage, we had to use interpolation methods to supply missing values. Firstly, we defined three functions for interpolation. 


```python
def MovingAVG(array, winsize):                                                                                # For computing moving average
  avg = []
  for i in range(len(array) - winsize + 1):
    avg.append(np.sum(array[i:i+winsize]) / winsize)
    
  avg = np.array(avg)
  
  return avg
  
def Interpolate_time_series(data):                                                                          # For interpolating the data using PCHIP                  
    x = np.arange(len(data))
    valid_indices = np.where(np.isfinite(data))
    filled_data = data.copy()
    filled_data[np.isnan(data)] = pchip_interpolate(x[valid_indices], data[valid_indices], x[np.isnan(data)])
    return filled_data
   
def process_n_return_distances(path, x_column, y_column, winsize, distance):                               # This function does interpolation and moving average, returns either Euclidian or 
                                                                                                              angular distance
  data = pd.read_csv(path)
  
  x = np.array(data[x_column])
  if np.sum(np.isnan(x))>0:
    x = Interpolate_time_series(x)
    
  x = MovingAVG(x , winsize)
  y = np.array(data[y_column])
  if np.sum(np.isnan(y))>0:
    y = Interpolate_time_series(y)
    
  y = MovingAVG(y , winsize)
  
  print('nans in original x:',np.sum(np.isnan(x)))
  print('nans in original y:',np.sum(np.isnan(y)))
  
  if distance == 'Euclidian':
    dist= np.sqrt( (x*x) + (y*y))
    
  elif distance == 'angle': 
    dist= np.arctan(y/x)
    
  return dist

```


Now we will proceed with converting the csv files into numpy files



```python
column = 'RAnkle' 
y_column = column+'_y'
x_column = column+'_x'
path='bodymovements'
  
dyads = range(1,24)
conds = ['FaNoOcc','FaOcc','NeNoOcc','NeOcc']
trials = range(1,4)
sbjs = range(2)
  
DICT = {}
for cnd in conds:
  for trl in range(1,4):
    for dyad in range(1,24):
      dyd_files = []
      for sbj in range(2):
        path_sub = path +'/'+'results_video_'+str(sbj)+'_'+cnd+'_'+str(trl)+'_pose_body_unprocessed-'+str(dyad)+'.csv'
        dyd_files.append(path_sub)
          
      DICT['('+cnd+','+str(trl)+','+str(dyad)+','+column+')'] = dyd_files
          
        
comb_files = list(DICT.keys())
  

  
  
  
for KEY in tqdm(comb_files):
    
  data = []
  for File in DICT[KEY]:
    data.append(process_n_return_distances(File, x_column, y_column, 30, 'Euclidian'))
        
  data = np.array(data)
  data = data.T
  data = data[::6,:]
  np.save('/user/swarag/Koul_et_al/signals/('+cnd+','+str(trl)+','+str(dyad)+')~.npy',data)  # Save the data to a numpy file
  
```


Remaining steps are same, except that the vision condition(visual contact vs no visual contact) and proximity(near vs far) will be derived from variable "cnd". 
