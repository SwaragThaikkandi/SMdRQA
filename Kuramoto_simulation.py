import networkx as nx
from SMdRQA.cross_validation import nested_cv
from SMdRQA.cross_validation import feature_selection
from SMdRQA.Extract_from_RP import First_middle_last_sliding_windows_all_vars
from SMdRQA.Extract_from_RP import First_middle_last_sliding_windows
from SMdRQA.Extract_from_RP import First_middle_last_avg
from SMdRQA.Extract_from_RP import windowed_RP
from SMdRQA.Extract_from_RP import Whole_window
from SMdRQA.Extract_from_RP import Sliding_window
from SMdRQA.Extract_from_RP import Check_Int_Array
from SMdRQA.Extract_from_RP import Mode
from SMdRQA.RP_maker import RP_computer
# see https://github.com/fabridamicelli/kuramoto
from kuramoto import Kuramoto, plot_phase_coherence, plot_activity
import ast
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

# For installing the kuramoto package run: pip install kuramoto


# Define fun


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
    # noisy_signal = np.clip(noisy_signal, np.min(signal), np.max(signal))

    return noisy_signal

##########################################################################


# These are the range of SNR values
SNRu = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
np.random.seed(301)
random.seed(301)
for i in tqdm(range(10)):
    # Selecting a random size for the number of oscillators, between 3 and 6
    N = random.sample(range(3, 7), 1)[0]
    # Sampling the natural frequencies of the oscillators from a normal
    # distribution
    omega = np.random.normal(0, 1, size=N)
    # Computing the critical coupling strength for the population of
    # oscillators
    Ks = (np.max(omega) - np.min(omega))
    # Sampling the coupling strength from a uniform distribution from 0 to 2Ks
    # so
    K_sample = np.random.uniform(low=0, high=2 * Ks, size=10)
    # that there is equal probability for finding a system having coupling strength below and
    # above critical coupling strength
    # Selecting a random length for the time series
    rp_sizes = random.sample(range(15, 45), len(K_sample))
    # Defining an all-to-all connected graph
    graph_nx = nx.erdos_renyi_graph(n=N, p=1)
    # Converting connection matrix to numpy matrix
    graph = nx.to_numpy_array(graph_nx)
    # Looping over different coupling strengths
    for k2 in tqdm(range(len(K_sample))):
        K_used = K_sample[k2]
        model = Kuramoto(
            coupling=K_used,
            dt=0.01,
            T=rp_sizes[k2],
            n_nodes=N,
            natfreqs=omega)  # Defining the Kuramoto model, dt is the timestep
        # Activation matrix from running the model
        act_mat = model.run(adj_mat=graph)
        # Taking transpose to get data from each oscillator as a column
        THETA = act_mat.T
        # Loop over different signal to noise ratios
        for snr in tqdm(SNRu):
            signals = THETA
            # Add noise to the signals
            for j in range(N):
                signals[:, j] = add_noise(signals[:, j], snr)

            np.save('/user/swarag/Kuramoto/signals/(' +
                    str(snr) +
                    ',' +
                    str(N) +
                    ',' +
                    str(Ks) +
                    ',' +
                    str(K_used) +
                    ',' +
                    str(rp_sizes[k2]) +
                    ')-.npy', signals)  # Save the data to a numpy file


# Gen
# directory to which the signals are saved
input_path = '/user/swarag/Kuramoto/signals'
# directory to which we want to save the RPs
RP_dir = '/user/swarag/Kuramoto/RP'
# generating RPs and saving to the specified folder
RP_computer(input_path, RP_dir)
# Ext
# Specifying window size and folder in which RPs are saved
Dict_RPs = windowed_RP(68, 'RP')
# Saving RQA variables to a csv file
First_middle_last_sliding_windows_all_vars(Dict_RPs, 'Kuramoto_data.csv')
# Pro
data = pd.read_csv('Kuramoto_data.csv')
# In the output data, the field named 'group' will have file name which
# contains details
FILE = np.array(data['group'])
SNR = []
NUM = []
Kc = []
K = []
SIZE = []
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
# Defining synchrony condition, coupling strength grater than that of the
# critical coupling strength
SYNCH = 1 * (K > Kc)
data['synch'] = SYNCH
# Select
data_ = data[data['snr'] == 1.0].reset_index(drop=True)
# Scale the data
# #########################################################################################################\
features = ['recc_rate',
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
    data_[feature] = (arr - np.mean(arr)) / (np.std(arr) + 10**(-9))

# Run the
nested_cv(
    data_,
    features,
    'synch',
    'Kuramotot(SNR=1.0)',
    repeats=100,
    inner_repeats=10,
    outer_splits=3,
    inner_splits=2)
# DONE !
