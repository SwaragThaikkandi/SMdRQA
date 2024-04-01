#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import log_loss

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

import os
import pickle
import pandas as pd
import seaborn as sns
from random import sample
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


def HistogramSampling(histogram, bins_midpoints, samples):
    '''\n    Function for doing histogram sampling from a distribution\n\n    Parameters\n    ----------\n\n    histogram : array\n        histogram counts normalised to get a sum =1\n\n      bins_midpoints : array\n          values corresponding to the probability values given as "histogram"\n\n      samples : int\n          number of histograms samples\n\n      Returns\n      -------\n\n      hist2 : array\n           histogram counts of the resulting distribution\n\n      bins_midpoints : array\n           midpoints of bins corresponding to hist2\n\n      References\n      ----------\n      - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n      '''

    cdf = np.cumsum(histogram)
    cdf = cdf / cdf[-1]
    values = np.random.rand(samples)
    value_bins = np.searchsorted(cdf, values)

    # print('values:',values)
    # print('value bins:',value_bins)

    random_from_cdf = bins_midpoints[value_bins]
    edges = list(bins_midpoints - 0.5)
    edges.append(bins_midpoints[-1] + 0.5)

    (hist2, edges1) = np.histogram(random_from_cdf, bins=edges)

    # print('bin_mids:',(edges1[:-1]+edges1[1:])/2)

    return (random_from_cdf, hist2)


# for percenthage laminarity

def Sliding_windowBS_sub_per_lam(
    RP,
    maxsize,
    winsize,
    n_boot,
):
    '''\n    Function for computing the CI of the bootstrapped confidence interval for percentage laminarity\n\n    Parameters\n    ----------\n\n    RP : ndarray\n        recurrence plot\n\n    maxsize : int\n        RP size\n\n    winsize : int\n        size of the sliding window\n\n     n_boot : int\n        number of bootstrap samples\n\n    Returns\n    -------\n\n    CI : double\n        confidence interval(95% quantile - 5% quantile)\n\n\n    References\n    ----------\n    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n    '''

    num_windows = maxsize - winsize + 1
    P = []
    P_bar = []
    for i in range(num_windows):
        sub_window = RP[i:i + winsize, i:i + winsize]
        diag_his = vert_hist(sub_window, winsize)

        # diag_his=diag_his[1:]

        P.append(diag_his / np.sum(diag_his))
        P_bar.append(diag_his)
    P = np.array(P)
    n_bar = int(np.sum(P_bar) / num_windows)
    print('n_bar:', n_bar)

    # print(np.sum(P, axis=1))

    P_summed = np.sum(P, axis=0)
    P_summed = P_summed / np.sum(P_summed)

    # print('n_bar:',n_bar)

    bins_midpoints = np.array(range(1, len(P_summed) + 1))

    # line_array=IND_array+1

    VAR = []
    for j in range(n_boot):
        (hist_sample, hist) = HistogramSampling(P_summed,
                                                bins_midpoints, n_bar)
        hist = hist / np.sum(hist)

        # percent_more=entropy(hist,2, winsize)

        percent_more = np.sum(hist[1:]) / np.sum(hist)

        # avg_line=entropy(sel_P,1,len(sel_P)-1)
        # print(avg_line)
        # print(avg_line)

        VAR.append(percent_more)

    # return np.std(VAR)**2

    return np.quantile(VAR, 0.95) - np.quantile(VAR, 0.05)


def Sliding_windowBS_sub_avg_vert(
    RP,
    maxsize,
    winsize,
    n_boot,
):
    '''\n    Function for computing the CI of the bootstrapped confidence interval for average vertical line length\n\n    Parameters\n    ----------\n\n    RP : ndarray\n        recurrence plot\n\n    maxsize : int\n        RP size\n\n    winsize : int\n        size of the sliding window\n\n     n_boot : int\n        number of bootstrap samples\n\n    Returns\n    -------\n\n    CI : double\n        confidence interval(95% quantile - 5% quantile)\n\n\n    References\n    ----------\n    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n    '''

    num_windows = maxsize - winsize + 1
    P = []
    P_bar = []
    for i in range(num_windows):
        sub_window = RP[i:i + winsize, i:i + winsize]
        diag_his = vert_hist(sub_window, winsize)

        # diag_his=diag_his[1:]

        P.append(diag_his / np.sum(diag_his))
        P_bar.append(diag_his)
    P = np.array(P)
    n_bar = int(np.sum(np.mean(P_bar, axis=0)))
    print('n_bar:', n_bar)

    # print(np.sum(P, axis=1))

    P_summed = np.sum(P, axis=0)
    P_summed = P_summed / np.sum(P_summed)

    # print('n_bar:',n_bar)

    bins_midpoints = np.array(range(1, len(P_summed) + 1))

    # line_array=IND_array+1

    VAR = []
    for j in range(n_boot):
        (hist_sample, hist) = HistogramSampling(P_summed,
                                                bins_midpoints, n_bar)
        hist = hist / np.sum(hist)

        # percent_more=entropy(hist,2, winsize)

        percent_more = np.mean(hist_sample[1:])

        # avg_line=entropy(sel_P,1,len(sel_P)-1)
        # print(avg_line)
        # print(avg_line)

        VAR.append(percent_more)

    # return np.std(VAR)**2

    return np.quantile(VAR, 0.95) - np.quantile(VAR, 0.05)


def Sliding_windowBS_sub_percent_det(
    RP,
    maxsize,
    winsize,
    n_boot,
):
    '''\n    Function for computing the CI of the bootstrapped confidence interval for percentage determinism\n\n    Parameters\n    ----------\n\n    RP : ndarray\n        recurrence plot\n\n    maxsize : int\n        RP size\n\n    winsize : int\n        size of the sliding window\n\n     n_boot : int\n        number of bootstrap samples\n\n    Returns\n    -------\n\n    CI : double\n        confidence interval(95% quantile - 5% quantile)\n\n\n    References\n    ----------\n    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n    '''

    num_windows = maxsize - winsize + 1
    P = []
    P_bar = []
    for i in range(num_windows):
        sub_window = RP[i:i + winsize, i:i + winsize]
        diag_his = diaghist(sub_window, winsize)

        # diag_his=diag_his[1:]

        P.append(diag_his / np.sum(diag_his))
        P_bar.append(diag_his)
    P = np.array(P)
    n_bar = int(np.sum(P_bar) / num_windows)
    print('n_bar:', n_bar)

    # print(np.sum(P, axis=1))

    P_summed = np.sum(P, axis=0)
    P_summed = P_summed / np.sum(P_summed)

    # print('n_bar:',n_bar)

    bins_midpoints = np.array(range(1, len(P_summed) + 1))

    # line_array=IND_array+1

    VAR = []
    for j in range(n_boot):
        (hist_sample, hist) = HistogramSampling(P_summed,
                                                bins_midpoints, n_bar)
        hist = hist / np.sum(hist)

        # percent_more=entropy(hist,2, winsize)

        percent_more = np.sum(hist[1:]) / np.sum(hist)

        # avg_line=entropy(sel_P,1,len(sel_P)-1)
        # print(avg_line)
        # print(avg_line)

        VAR.append(percent_more)

    # return np.std(VAR)**2

    return np.quantile(VAR, 0.95) - np.quantile(VAR, 0.05)


def Sliding_windowBS_sub_avg_diag(
    RP,
    maxsize,
    winsize,
    n_boot,
):
    '''\n    Function for computing the CI of the bootstrapped confidence interval for average diagonal line length\n\n    Parameters\n    ----------\n\n    RP : ndarray\n        recurrence plot\n\n    maxsize : int\n        RP size\n\n    winsize : int\n        size of the sliding window\n\n     n_boot : int\n        number of bootstrap samples\n\n    Returns\n    -------\n\n    CI : double\n        confidence interval(95% quantile - 5% quantile)\n\n\n    References\n    ----------\n    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n    '''

    num_windows = maxsize - winsize + 1
    P = []
    P_bar = []
    for i in range(num_windows):
        sub_window = RP[i:i + winsize, i:i + winsize]
        diag_his = diaghist(sub_window, winsize)

        # diag_his=diag_his[1:]

        P.append(diag_his / np.sum(diag_his))
        P_bar.append(diag_his)
    P = np.array(P)
    n_bar = int(np.sum(np.mean(P_bar, axis=0)))
    print('n_bar:', n_bar)

    # print(np.sum(P, axis=1))

    P_summed = np.sum(P, axis=0)
    P_summed = P_summed / np.sum(P_summed)

    # print('n_bar:',n_bar)

    bins_midpoints = np.array(range(1, len(P_summed) + 1))

    # line_array=IND_array+1

    VAR = []
    for j in range(n_boot):
        (hist_sample, hist) = HistogramSampling(P_summed,
                                                bins_midpoints, n_bar)
        hist = hist / np.sum(hist)

        # percent_more=entropy(hist,2, winsize)

        percent_more = np.mean(hist_sample[1:])

        # avg_line=entropy(sel_P,1,len(sel_P)-1)
        # print(avg_line)
        # print(avg_line)

        VAR.append(percent_more)

    # return np.std(VAR)**2

    return np.quantile(VAR, 0.95) - np.quantile(VAR, 0.05)


def Sliding_windowBS(
    RP,
    maxsize,
    var,
    n_boot=1000,
):
    '''\n    Function for computing the CI of the bootstrapped confidence interval for a given variable for different window sizes starting from 20 to a maximum of RP size incremented by value of 10\n\n    Parameters\n    ----------\n\n    RP : ndarray\n        recurrence plot\n\n    maxsize : int\n        RP size\n\n    var : str\n        RQA variable name\n\n        *percent_lam* : percentage laminarity\n\n        *percent_det* : percentage determinism\n\n        *avg_vert* : average value of vertical line distribution\n\n        *avg_diag* : average value of diagonal line distribution\n\n     n_boot : int\n        number of bootstrap samples\n\n    Returns\n    -------\n\n    data : dataframe\n        dataframe containing the CI estimate of corresponding window sizes, columns are\n\n        *WINSIZE* : window size\n\n        *95% quantile- 5% quantile* : confidence interval\n\n\n    References\n    ----------\n    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n    '''

    WINSIZE = []
    VAR = []
    for i in tqdm(range(20, maxsize, 10)):
        if var == 'percent_lam':
            var_std = Sliding_windowBS_sub_per_lam(RP, maxsize, i,
                                                   n_boot)
        elif var == 'percent_det':
            var_std = Sliding_windowBS_sub_percent_det(RP, maxsize, i,
                                                       n_boot)
        elif var == 'avg_vert':
            var_std = Sliding_windowBS_sub_avg_vert(RP, maxsize, i,
                                                    n_boot)
        elif var == 'avg_diag':
            var_std = Sliding_windowBS_sub_avg_diag(RP, maxsize, i,
                                                    n_boot)

        WINSIZE.append(i)
        VAR.append(var_std)

    DICT = {'WINSIZE': WINSIZE, '95% quantile- 5% quantile': VAR}
    df = pd.DataFrame.from_dict(DICT)
    return df


def Sliding_window_whole_data(RP_dir, var, n_boot=1000):
    '''\n    Function for computing the CI of the bootstrapped confidence interval for a given variable for different window sizes starting from 20 to a maximum of RP size incremented by value of 10. Done for all datasets in the folder specified\n\n    Parameters\n    ----------\n\n    RP_dir : str\n        directory containing all the RP files(.npy)\n\n    var : str\n        RQA variable name\n\n        *percent_lam* : percentage laminarity\n\n        *percent_det* : percentage determinism\n\n        *avg_vert* : average value of vertical line distribution\n\n        *avg_diag* : average value of diagonal line distribution\n\n     n_boot : int\n        number of bootstrap samples\n\n    Returns\n    -------\n\n    data : dataframe\n        dataframe containing the CI estimate of corresponding window sizes, columns are\n\n        *WINSIZE* : window size\n\n        *95% quantile- 5% quantile* : confidence interval\n\n        *group* : filename of the RPs\n\n\n    References\n    ----------\n    - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237\xe2\x80\x93329.\n\n    '''

    files = os.listdir(RP_dir)
    DFs = []
    for File in files:
        path = RP_dir + '/' + File
        info = File.split('-')[0]
        RP = np.load(path)
        maxsize = len(RP)
        df = Sliding_windowBS(RP, maxsize, var, n_boot=n_boot)
        df['group'] = info
        DFs.append(df)

    df_out = pd.concat(DFs)

    return df_out
