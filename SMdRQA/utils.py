from scipy.spatial import distance_matrix
from scipy.special import digamma
from scipy.spatial import distance
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


def assert_matrix(input_array):
    '''
    Function to assert that an input is a matrix

    Parameters
    ----------
    input_array   : ndarray
        double array

    Returns
    -------

    matrix : ndarray
         2D matrix

    '''
    # Check if the input is a NumPy array
    if not isinstance(input_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Ensure the array is at least 2D (convert to matrix if necessary)
    matrix = np.atleast_2d(input_array)

    return matrix


def compute_3D_matrix_size(dim1, dim2, dim3, dtype=np.float64):
    '''
    Function to calculate memory required(GiB) for storing a 3D matrix of given size and data type

    Parameters
    ----------
    dim1   : int
        dimension

    dim2   : int
        dimension

    dim3   : int
        dimension

    dtype   : dtype
        data type, default = np.float64

    Returns
    -------

    size : double
         size of the 3D matrix in GiB

    '''
    total_elements = dim1 * dim2 * dim3
    element_size = np.dtype(dtype).itemsize
    return (total_elements * element_size) / (1024**3)


def assert_3D_matrix_size(dim1, dim2, dim3, dtype=np.float64, memory_limit=4):
    '''
    Function to assert that a 3D matrix of given dimensions and data type is below a specified size

    Parameters
    ----------
    dim1   : int
        dimension

    dim2   : int
        dimension

    dim3   : int
        dimension

    dtype   : dtype
        data type, default = np.float64

    memory_limit   : double
        memory limit in GiB, default = 4

    Returns
    -------

    output : bool
         True if the memory requirement is less than the specified limit

    '''
    req_memory = compute_3D_matrix_size(dim1, dim2, dim3, dtype=np.float64)

    if req_memory < memory_limit:
        return True
    elif req_memory >= memory_limit:
        return False
