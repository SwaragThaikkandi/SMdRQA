import seaborn as sns
from SMdRQA.RQA_functions import entropy
from SMdRQA.RQA_functions import average
from SMdRQA.RQA_functions import maxi
from SMdRQA.RQA_functions import mode
from SMdRQA.RQA_functions import percentmorethan
from SMdRQA.RQA_functions import diaghist
from SMdRQA.RQA_functions import onedhist
from SMdRQA.RQA_functions import vert_hist
from SMdRQA.RQA_functions import plotwindow
from SMdRQA.RQA_functions import findeps
from SMdRQA.RQA_functions import reccrate
from SMdRQA.RQA_functions import reccplot
from SMdRQA.RQA_functions import findm
from SMdRQA.RQA_functions import fnnhitszero
from SMdRQA.RQA_functions import fnnratio
from SMdRQA.RQA_functions import nearest
from SMdRQA.RQA_functions import delayseries
from SMdRQA.RQA_functions import findtau
from SMdRQA.RQA_functions import timedelayMI
from SMdRQA.RQA_functions import mutualinfo
from SMdRQA.RQA_functions import binscalc
from SMdRQA.RQA_functions import doanes_formula
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


def fnnhitszero_Plot(u, n, d, m, tau, sig, delta, Rmin, Rmax, rdiv):
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

    Rarr : array
       an array of r values

    FNN : array
       corresponding false nearest neighbour values

    References
    ----------
    - Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45 (6), 3403.

    '''
    Rarr = np.linspace(Rmin, Rmax, rdiv)
    FNN = []
    for i in range(rdiv):
        FNN.append(fnnratio(u, n, d, m, tau, Rarr[i], sig))

    return Rarr, FNN


def findm_Plot(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound, save_path):
    '''
    This is an effort to make plot given in Kantz, & Schreiber(2004) section 3.3.1

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
         bound value for terminating the parameter serch for m

    Returns
    -------

    Saves a figure and a pickle file

    References
    ----------
    - Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45 (6), 3403.

    - Kantz, H., & Schreiber, T. (2004). Nonlinear time series analysis (Vol. 7). Cambridge university press. section 3.3.1

    '''
    plot = True
    if plot:
        mmax = int((3 * d + 11) / 2)

        DICT = {}
        plt.figure(figsize=(16, 12))
        for m in range(1, mmax):
            color = plt.cm.rainbow(m / (mmax - 1))
            Rarr, FNN = fnnhitszero_Plot(
                u, n, d, mmax - m, tau, sd, delta, Rmin, Rmax, rdiv)
            label = 'm=' + str(mmax - m)
            plt.plot(Rarr, FNN, color=color, label=label)
            DICT[mmax - m] = {'r': Rarr, 'FNN': FNN}

        plt.xlabel('r')
        plt.ylabel('FNN')
        plt.title('r vs FNN plot')
        plt.legend()
        plt.savefig(save_path + '.pdf')
        with open(save_path + '.pickle', 'wb') as handle:
            pickle.dump(DICT, handle)


def RP_diagnose(
        input_path,
        diagnose_dir,
        rdiv=451,
        Rmin=1,
        Rmax=10,
        delta=0.001,
        bound=0.2):
    '''
    Function to diagnose issues in finding the embedding dimension. It is similar to RP maker, but it deos not generate RP, nstead saves r vs FNN plot varying embedding dimensions and such plots are saved for each of the time series files present in the input directory

    Parameters
    ----------

    input_path : str
         folder containing the numpy files, rows> number of samples, columns> number of streams

    diagnose_dir : str
        folder where the plots needed for checks should be saved

    rdiv       : int
         number of divisions(resolution) for the variable r during parameter search for embedding dimension

    Rmax       : double
         maximum value for the variable r during parameter search for embedding dimension

    Rmin       : double
         minimum value for the variable r during parameter search for embedding dimension

    delta      : double
         the tolerance value below which an FNN value will be considered as zero

    bound      : double
         This is the value in the r value(at which FNN hits zero) va embedding dimension plot. The search is terminated if the value goes below this tolerance value and the value just below tolerance value is reported for embedding dimmension


    Returns
    -------

    Saves r vs FNN plot varying embedding dimensions and such plots are saved for each of the time series files present in the input directory to a path specified as diagnose directory

    Error_Report_Sheet : file
            This is a csv file containing details of the files for which RP calculation was failed because of numpy.core._exceptions.MemoryError. This is due to an issue at the time delay estimation part, check dimensionality of the data


    References
    ----------

    - Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45 (6), 3403.

    - Kantz, H., & Schreiber, T. (2004). Nonlinear time series analysis (Vol. 7). Cambridge university press. section 3.3.1

    '''
    path = input_path
    files = os.listdir(path)
    ERROROCC = []
    FILE = []
    TAU = []
    Marr = []
    EPS = []
    BOUND = []

    for File in tqdm(files):
        try:
            file_path = path + '/' + File
            data = np.load(file_path)
            (M, N) = data.shape

            data = (data - np.mean(data, axis=0, keepdims=True)) / \
                np.std(data, axis=0, keepdims=True)

            n = M
            d = N
            u = data

            sd = 3 * np.std(u)
            print('starting tau calculation ...')
            tau = findtau(u, n, d, 0)
            print('Done Tau calculation....')
            print('TAU:', tau)
            print('starting m calculation ...')
            # notFound = 1
            # while notFound == 1:
        # try:
            File_out = File.split('.')[0]
            try:
                findm_Plot(
                    u,
                    n,
                    d,
                    tau,
                    sd,
                    delta,
                    Rmin,
                    Rmax,
                    rdiv,
                    bound,
                    diagnose_dir +
                    '/' +
                    File_out)

            except ValueError:
                print('unable to compute due to value error')
                ERROROCC.append(File)

        except MemoryError:
            print("Couldn't do computation due to numpy.core._exceptions.MemoryError")
            ERROROCC.append(File)

    DICT = {'error occurances': ERROROCC}
    df_out = pd.DataFrame.from_dict(DICT)
    df_out.to_csv('Error_Report_Sheet.csv')


def get_minFNN_distribution_plot(path, save_name):
    '''
    This is a function used to get the delta value or the value used to effectively consider
    a particular value of FNN effectively as zero. This function estimates and plots the distribution
    of minimum FNN value for different embedding dimensions(m). It computes the upper(2.5% quantile)
    and lower(97.5% quantile) and when we want to get r(at FNN hitting zero) vs m graph, generally the
    delta value should be more than the highest upper bound(most likely for m=1) is choosen

    Parameters
    ----------
    path   : str
        path to folder containing pickes files computed using "RP_diagnose" function

    savename : str
         The output plot and CSV file would be saved in the name specified

    Returns
    -------

    Saves a plot and a CSV file

    References
    ----------
    - Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45 (6), 3403.

    '''
    files = os.listdir(path)
    M = []
    FILE = []
    MIN_fnn = []

    for File in files:
        open_path = path + '/' + File
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

    DICT = {'file': FILE, 'm': M, 'min_fnn': MIN_fnn}
    df_out = pd.DataFrame.from_dict(DICT)
    plt.figure(figsize=(12, 9))
    sns.violinplot(data=df_out, x='m', y='min_fnn')
    plt.savefig(save_name + '.png')
    Ms = np.unique(np.array(df_out['m']))
    low_b = []
    up_b = []
    for m in Ms:
        df_out_sub = df_out[df_out['m'] == m]
        mfnn = np.array(df_out_sub['min_fnn'])
        low_b.append(np.quantile(mfnn, 0.025))
        up_b.append(np.quantile(mfnn, 0.975))

    DICT2 = {'m': Ms, '2.5% quantile': low_b, '97.5% quantile': up_b}
    df_out2 = pd.DataFrame.from_dict(DICT2)
    df_out2.to_csv(save_name + '.csv')
