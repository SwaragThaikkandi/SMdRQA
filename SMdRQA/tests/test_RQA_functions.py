from SMdRQA.RQA_functions import *
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


def test_findtau():
    SIZE = 10
    np.random.seed(seed=301)
    angle = np.linspace(0, 64 * np.pi, 9600)
    n = int(((1200) / 4) * (12 - SIZE))
    d = 1
    angle = angle[0:n]
    u = np.zeros((n, d))
    var = np.sin((2 * np.pi * np.random.uniform(0, 1)) + (4 * angle))
    u[:, 0] = (var - np.mean(var)) / np.std(var)
    sd = 3 * np.std(u)
    tau = findtau(u, n, d, '0')
    assert ((tau > 0) and (tau < n))


def test_KNN_MI():
    num_rows = 1000  # Number of time points (rows)
    num_cols = 2     # Number of sine waves (columns)

    # Create a time vector
    t = np.linspace(0, 2 * np.pi, num_rows)

    # Initialize an empty matrix
    sin_matrix = np.zeros((num_rows, num_cols))

    # Generate sine waves with different frequencies
    # Example frequencies for each sine wave
    frequencies = np.linspace(1, 5, num_cols)

    for i in range(num_cols):
        sin_matrix[:, i] = np.sin(frequencies[i] * t)

    mi1 = KNN_MI_vectorized(sin_matrix, sin_matrix, 5)
    mi2 = KNN_MI_non_vectorized(sin_matrix, sin_matrix, 5)
    print('mi1:', mi1)
    print('mi2:', mi2)
    assert abs(mi1 - mi2) < 0.01


def test_findm():
    SIZE = 10
    rdiv = 451
    Rmin = 1
    Rmax = 10

    delta = 0.001
    bound = 0.2
    np.random.seed(seed=301)
    angle = np.linspace(0, 64 * np.pi, 9600)
    n = int(((1200) / 4) * (12 - SIZE))
    d = 1
    angle = angle[0:n]
    u = np.zeros((n, d))
    var = np.sin((2 * np.pi * np.random.uniform(0, 1)) + (4 * angle))
    u[:, 0] = (var - np.mean(var)) / np.std(var)
    sd = 3 * np.std(u)
    tau = findtau(u, n, d, 0)
    m = findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound)
    assert m > 0


def test_findeps():
    SIZE = 10
    rdiv = 451
    Rmin = 1
    Rmax = 10

    delta = 0.001
    bound = 0.2
    reqrr = 0.1
    rr_delta = 0.005
    epsmin = 0
    epsmax = 10
    epsdiv = 1001
    windnumb = 1
    np.random.seed(seed=301)
    angle = np.linspace(0, 64 * np.pi, 9600)
    n = int(((1200) / 4) * (12 - SIZE))
    d = 1
    angle = angle[0:n]
    u = np.zeros((n, d))
    var = np.sin((2 * np.pi * np.random.uniform(0, 1)) + (4 * angle))
    u[:, 0] = (var - np.mean(var)) / np.std(var)
    sd = 3 * np.std(u)
    tau = findtau(u, n, d, 0)
    m = findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound)
    eps = findeps(u, n, d, m, tau, reqrr, rr_delta, epsmin, epsmax, epsdiv)
    assert eps > 0


def test_recc_plot():
    SIZE = 10
    rdiv = 451
    Rmin = 1
    Rmax = 10

    delta = 0.001
    bound = 0.2
    reqrr = 0.1
    rr_delta = 0.005
    epsmin = 0
    epsmax = 10
    epsdiv = 1001
    windnumb = 1
    np.random.seed(seed=301)
    angle = np.linspace(0, 64 * np.pi, 9600)
    n = int(((1200) / 4) * (12 - SIZE))
    d = 1
    angle = angle[0:n]
    u = np.zeros((n, d))
    var = np.sin((2 * np.pi * np.random.uniform(0, 1)) + (4 * angle))
    u[:, 0] = (var - np.mean(var)) / np.std(var)
    sd = 3 * np.std(u)
    tau = findtau(u, n, d, 0)
    m = findm(u, n, d, tau, sd, delta, Rmin, Rmax, rdiv, bound)
    eps = findeps(u, n, d, m, tau, reqrr, rr_delta, epsmin, epsmax, epsdiv)
    rplot = reccplot(u, n, d, m, tau, eps)
    (M, N) = rplot.shape

    assert M == N
    assert M > 0
