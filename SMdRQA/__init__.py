#!/usr/bin/python

# see https://github.com/fabridamicelli/kuramoto
from kuramoto import Kuramoto, plot_phase_coherence, plot_activity
import numba
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
__docformat__ = "restructuredtext"

__version__ = "2024.04.01"

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
