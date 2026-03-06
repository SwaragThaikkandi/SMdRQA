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
from SMdRQA.RQA_functions import findeps_multi
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


def RP_computer(
        input_path,
        RP_dir,
        rdiv=451,
        Rmin=1,
        Rmax=10,
        delta=0.001,
        bound=0.2,
        reqrr=0.1,
        rr_delta=0.005,
        epsmin=0,
        epsmax=10,
        epsdiv=1001,
        windnumb=1,
        group_level=False,
        group_level_estimates=None):
    '''
      Function to compute diagonal line distribution(counts of line lengths)

      Parameters
      ----------

      input_path : str
          folder containing the numpy files, rows> number of samples, columns> number of streams

      RP_dir     : str
          directory in which the RPs should be stored

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

      req_rr     : double
           This is a variable that user can define. This controls the overall recurrence rate of the whole RP

      rr_delta   : double
           This variable is used to define tolerance for accepting a value of neighbourhood radius. If the absolte differenece between the resccurence rate value to that of the desired value is less than this tolerance, the value of epsilon is accepted

      eps_min    : double
           Minimum value of the neighbourhood radius value to begin with for fixing the reccurrence rate

      eps_max    : double
           Maximum value of the neighbourhood radius value above which the search won't progress

      eps_div    : double
           Number of divisions between eps_min and eps_max

      group_level: boolean
           Whether to estimate some variables at the group level and keep them fixed across RPs.

      group_level_estimates: list
           List of variables needed to estimate at the group level. Applicable only if "group_level = True".
           List elements should be like: ['eps', 'm'], ['eps'], ['eps', 'tau']
           - 'eps' : neighbourhood radius
           - 'm' : embedding dimension
           - 'tau' : time delay

      Returns
      -------

      Saves RPs for each of the signal present in the input directory. Additionally, in your root directory check for following files

      Error_Report_Sheet : file
             This is a csv file containing details of the files for which RP calculation was failed because of numpy.core._exceptions.MemoryError. This is due to an issue at the time delay estimation part, check dimensionality of the data

      param_Sheet        : file
             The RQA parameter values for those signals for which the RPs were computed without any fail

      References
      ----------
      - Webber Jr, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of applied physiology, 76 (2), 965–973.
      - Webber Jr, C. L., & Zbilut, J. P. (2005). Recurrence quantification analysis of nonlinear dynamical systems. Tutorials in contemporary nonlinear methods for the behavioral sciences, 94 (2005), 26–94.
      - Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics reports, 438 (5-6), 237–329.
      - Marwan, N., Schinkel, S., & Kurths, J. (2013). Recurrence plots 25 years later—gaining confidence in dynamical transitions. Europhysics Letters, 101 (2), 20007.
      - Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence- plot-based measures of complexity and their application to heart-rate-variability data. Physical review E, 66 (2), 026702.

      '''
    if not group_level:
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
                try:
                    m = findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound)
                    print('Done m calculation....')
                    print('m:', m)
                    print('starting eps calculation ...')
                    eps = findeps(
                        u,
                        n,
                        d,
                        m,
                        tau,
                        reqrr,
                        rr_delta,
                        epsmin,
                        epsmax,
                        epsdiv)
                    print('Done eps calculation....')
                    print('EPS:', eps)
                    rplot = reccplot(u, n, d, m, tau, eps)
                    print('Done rplot calculation....')
                    rplotwind = rplot
                    np.save(RP_dir + '/' + File, rplotwind)

                    # notFound = 0
                    FILE.append(File)
                    TAU.append(tau)
                    Marr.append(m)
                    EPS.append(eps)
                    BOUND.append(bound)

                except ValueError:
                    print('unable to compute due to value error')
                    ERROROCC.append(File)

            except MemoryError:
                print(
                    "Couldn't do computation due to numpy.core._exceptions.MemoryError")
                ERROROCC.append(File)

        DICT = {'error occurances': ERROROCC}
        df_out = pd.DataFrame.from_dict(DICT)
        df_out.to_csv('Error_Report_Sheet.csv')

        DICT2 = {
            'file': FILE,
            'tau': TAU,
            'm': Marr,
            'eps': EPS,
            'bound': BOUND}
        df_out2 = pd.DataFrame.from_dict(DICT2)
        df_out2.to_csv('param_Sheet.csv')

    elif group_level:
        path = input_path
        files = os.listdir(path)
        ERROROCC = []
        FILE = []
        TAU = []
        Marr = []
        EPS = []
        BOUND = []
        U_arr = []
        N_arr = []
        D_arr = []

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
                U_arr.append(u)
                N_arr.append(n)
                D_arr.append(d)
                sd = 3 * np.std(u)
                print('starting tau calculation ...')
                tau = findtau(u, n, d, 0)
                print('Done Tau calculation....')
                print('TAU:', tau)
                print('starting m calculation ...')
                # notFound = 1
                # while notFound == 1:
                # try:
                try:
                    m = findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound)
                    print('Done m calculation....')
                    print('m:', m)
                    print('starting eps calculation ...')
                    eps = findeps(
                        u,
                        n,
                        d,
                        m,
                        tau,
                        reqrr,
                        rr_delta,
                        epsmin,
                        epsmax,
                        epsdiv)
                    print('Done eps calculation....')
                    print('EPS:', eps)

                    print('Done rplot calculation....')

                    # notFound = 0
                    FILE.append(File)
                    TAU.append(tau)
                    Marr.append(m)
                    EPS.append(eps)
                    BOUND.append(bound)

                except ValueError:
                    print('unable to compute due to value error')
                    ERROROCC.append(File)

            except MemoryError:
                print(
                    "Couldn't do computation due to numpy.core._exceptions.MemoryError")
                ERROROCC.append(File)

        DICT = {'error occurances': ERROROCC}
        df_out = pd.DataFrame.from_dict(DICT)
        df_out.to_csv('Error_Report_Sheet.csv')

        DICT2 = {
            'file': FILE,
            'tau': TAU,
            'm': Marr,
            'eps': EPS,
            'bound': BOUND}
        df_out2 = pd.DataFrame.from_dict(DICT2)
        df_out2.to_csv('param_Sheet.csv')
        eps_mean = findeps_multi(
            U_arr,
            N_arr,
            D_arr,
            Marr,
            TAU,
            reqrr,
            rr_delta,
            epsmin,
            epsmax,
            epsdiv)
        for i2 in range(len(FILE)):
            file_path = path + '/' + FILE[i2]
            data = np.load(file_path)
            (M, N) = data.shape

            data = (data - np.mean(data, axis=0, keepdims=True)) / \
                np.std(data, axis=0, keepdims=True)
            n = M
            d = N
            u = data

            if 'tau' in group_level_estimates:
                tau_ = np.mean(TAU)
            else:
                tau_ = TAU[i2]
            #
            if 'm' in group_level_estimates:
                m_ = np.mean(Marr)
            else:
                m_ = Marr[i2]
            #
            if 'eps' in group_level_estimates:
                eps_ = eps_mean
            else:
                eps_ = EPS[i2]
            #
            rplot = reccplot(u, n, d, m_, tau_, eps_)
            rplotwind = rplot
            np.save(RP_dir + '/' + FILE[i2], rplotwind)
