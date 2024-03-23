# Kuramoto Model
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

``python
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
