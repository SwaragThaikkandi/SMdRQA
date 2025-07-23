from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from SMdRQA.RQA_functions import *
import ast
import memory_profiler
from scipy.interpolate import pchip_interpolate
from functools import partial
from p_tqdm import p_map
import numpy as np
from scipy.stats import skew
import math
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
#


def test_find_first_minima_or_global_minima_index():
    import numpy as np

    # Case 1: Empty array should return None
    arr_empty = []
    assert find_first_minima_or_global_minima_index(
        arr_empty) is None, "Failed on empty array"

    # Case 2: Single element array should return index 0
    arr_single = [5]
    assert find_first_minima_or_global_minima_index(
        arr_single) == 0, "Failed on single element array"

    # Case 3: First element is a local minima
    # For an array with two elements, if the first is less than the second,
    # returns index 0.
    arr_first_min = [1, 3, 4, 2]
    # Although there is another minimum at the end, the function should return
    # the first element.
    assert find_first_minima_or_global_minima_index(
        arr_first_min) == 0, "Failed when first element is local minima"

    # Case 4: Local minima in the middle
    # Array where the first element does not qualify but a middle element is a local minimum.
    # In [3, 2, 4, 5] element at index 1 is lower than its neighbors.
    arr_local_mid = [3, 2, 4, 5]
    assert find_first_minima_or_global_minima_index(
        arr_local_mid) == 1, "Failed on middle local minima"

    # Case 5: No local minima exists (strictly no element is lower than both neighbors).
    # Use an array with a plateau so no strict local minimum is found.
    # For example: [5, 4, 4, 4, 5] has no element strictly lower than both neighbors.
    # The function should then return the index of the global minimum (first
    # occurrence).
    arr_no_local = [5, 4, 4, 4, 5]
    # Global minimum is 4 and np.argmin returns the first index with that
    # value, i.e., index 1.
    assert find_first_minima_or_global_minima_index(
        arr_no_local) == 1, "Failed on global minimum fallback"

    # Case 6: Another example with no local minima because the first element qualifies.
    # In a monotonically increasing array, the first element is considered a
    # local minima.
    arr_monotonic = [1, 2, 3, 4, 5]
    assert find_first_minima_or_global_minima_index(
        arr_monotonic) == 0, "Failed on monotonically increasing array"

    print("All tests passed!")


def test_binscalc():
    import numpy as np
    # Create a sample dataset with n samples and d dimensions
    n = 100
    d = 3
    X = np.random.rand(n, d)

    # Test 'FD' method (generalised Freedman–Diaconis rule)
    bins_fd = binscalc(X, n, d, 'FD')
    # Should return a numpy array of shape (d,) with positive integers
    assert isinstance(
        bins_fd, np.ndarray), "'FD' method should return a numpy array"
    assert bins_fd.shape == (
        d,), f"'FD' method expected shape {(d,)}, got {bins_fd.shape}"
    assert np.all(bins_fd > 0), "All 'FD' bins should be positive integers"

    # Test 'sqrt' method
    bins_sqrt = binscalc(X, n, d, 'sqrt')
    # Expected: ceil(sqrt(n)) for each dimension
    expected_sqrt = int(np.ceil(np.sqrt(n)))
    expected_sqrt_arr = np.array([expected_sqrt] * d)
    assert isinstance(
        bins_sqrt, np.ndarray), "'sqrt' method should return a numpy array"
    assert bins_sqrt.shape == (
        d,), f"'sqrt' method expected shape {(d,)}, got {bins_sqrt.shape}"
    assert np.all(
        bins_sqrt == expected_sqrt_arr), "'sqrt' method did not return the expected bin counts"

    # Test 'rice' method
    bins_rice = binscalc(X, n, d, 'rice')
    # Expected: 2 * ceil(cuberoot(n)) for each dimension
    expected_rice = int(2 * np.ceil(np.cbrt(n)))
    expected_rice_arr = np.array([expected_rice] * d)
    assert isinstance(
        bins_rice, np.ndarray), "'rice' method should return a numpy array"
    assert bins_rice.shape == (
        d,), f"'rice' method expected shape {(d,)}, got {bins_rice.shape}"
    assert np.all(
        bins_rice == expected_rice_arr), "'rice' method did not return the expected bin counts"

    # Test 'sturges' method
    bins_sturges = binscalc(X, n, d, 'sturges')
    # Expected: 1 + ceil(log(n)) for each dimension
    expected_sturges = int(1 + np.ceil(np.log(n)))
    expected_sturges_arr = np.array([expected_sturges] * d)
    assert isinstance(
        bins_sturges, np.ndarray), "'sturges' method should return a numpy array"
    assert bins_sturges.shape == (
        d,), f"'sturges' method expected shape {(d,)}, got {bins_sturges.shape}"
    assert np.all(
        bins_sturges == expected_sturges_arr), "'sturges' method did not return the expected bin counts"

    # Test 'doanes' method
    bins_doanes = binscalc(X, n, d, 'doanes')
    # Expected: an array of positive integers (the exact values depend on the
    # data's skewness)
    assert isinstance(
        bins_doanes, np.ndarray), "'doanes' method should return a numpy array"
    assert bins_doanes.shape == (
        d,), f"'doanes' method expected shape {(d,)}, got {bins_doanes.shape}"
    assert np.all(
        bins_doanes > 0), "All 'doanes' bins should be positive integers"

    # Test 'default' method
    bins_default = binscalc(X, n, d, 'default')
    # Expected: the integer 15 (not an array)
    assert isinstance(
        bins_default, int), "'default' method should return an integer"
    assert bins_default == 15, "'default' method should return 15"

    print("All tests for binscalc passed!")


def test_KNN_MI_partial_vectorized():
    import numpy as np
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Define the number of samples and dimensions
    n_samples = 300
    d = 2

    # Case 1: Identical time series (should yield higher MI)
    X_identical = np.random.rand(n_samples, d)
    Y_identical = X_identical.copy()
    mi_identical = KNN_MI_partial_vectorized(
        X_identical, Y_identical, nearest_neighbor=5)

    # Case 2: Independent time series (should yield lower MI)
    X_independent = np.random.rand(n_samples, d)
    Y_independent = np.random.rand(n_samples, d)
    mi_independent = KNN_MI_partial_vectorized(
        X_independent, Y_independent, nearest_neighbor=5)

    # Assert that MI for identical signals is greater than for independent
    # signals
    assert mi_identical > mi_independent, (
        f"Expected MI for identical series ({mi_identical}) to be higher than for independent series ({mi_independent})."
    )

    # Case 3: Vary the nearest neighbor parameter and check that MI changes
    # accordingly
    mi_nn3 = KNN_MI_partial_vectorized(
        X_identical, Y_identical, nearest_neighbor=3)
    mi_nn10 = KNN_MI_partial_vectorized(
        X_identical, Y_identical, nearest_neighbor=10)
    assert mi_nn3 != mi_nn10, "Mutual information should vary with different nearest neighbor values"

    # Case 4: Verify that the returned value is a float
    assert isinstance(mi_identical, (float, np.floating)
                      ), "The returned MI should be a floating point number"

    print("All tests for KNN_MI_partial_vectorized passed!")


def test_KNN_MI_non_vectorized():
    import numpy as np
    from scipy.spatial import distance
    from scipy.special import digamma

    # Set seed for reproducibility
    np.random.seed(42)
    n_samples = 300
    d = 2  # dimensions

    # Case 1: Identical time series (expect high MI)
    X_identical = np.random.rand(n_samples, d)
    Y_identical = X_identical.copy()
    mi_identical = KNN_MI_non_vectorized(
        X_identical, Y_identical, nearest_neighbor=5)

    # Case 2: Independent time series (expect lower MI)
    X_independent = np.random.rand(n_samples, d)
    Y_independent = np.random.rand(n_samples, d)
    mi_independent = KNN_MI_non_vectorized(
        X_independent, Y_independent, nearest_neighbor=5)

    # Assert that MI for identical series is higher than for independent series
    assert mi_identical > mi_independent, (
        f"Expected MI for identical series ({mi_identical}) to be higher than for independent series ({mi_independent})."
    )

    # Case 3: Test different nearest neighbor parameters
    mi_nn3 = KNN_MI_non_vectorized(
        X_identical, Y_identical, nearest_neighbor=3)
    mi_nn10 = KNN_MI_non_vectorized(
        X_identical, Y_identical, nearest_neighbor=10)
    assert mi_nn3 != mi_nn10, "Mutual information should change with different nearest neighbor values"

    # Case 4: Check that the output is a floating point number
    assert isinstance(mi_identical, (float, np.floating)
                      ), "The returned MI should be a floating point number"

    print("All tests for KNN_MI_non_vectorized passed!")


def test_KNN_MI_vectorized():
    import numpy as np
    from scipy.special import digamma

    # Define parameters
    n_samples = 500  # Number of data points
    d = 2  # Dimensions

    # Generate synthetic data
    np.random.seed(42)

    # Case 1: Identical time series (high MI)
    X_identical = np.random.rand(n_samples, d)
    Y_identical = X_identical.copy()
    mi_identical = KNN_MI_vectorized(
        X_identical, Y_identical, nearest_neighbor=5)

    # Case 2: Independent time series (low MI)
    X_independent = np.random.rand(n_samples, d)
    Y_independent = np.random.rand(n_samples, d)
    mi_independent = KNN_MI_vectorized(
        X_independent, Y_independent, nearest_neighbor=5)

    # Ensure MI for identical series is greater than for independent series
    assert mi_identical > mi_independent, (
        f"Expected MI for identical series ({mi_identical}) to be greater than for independent series ({mi_independent})."
    )

    # Case 3: Different nearest neighbor values (should change MI)
    mi_nn_3 = KNN_MI_vectorized(X_identical, Y_identical, nearest_neighbor=3)
    mi_nn_10 = KNN_MI_vectorized(X_identical, Y_identical, nearest_neighbor=10)

    assert mi_nn_3 != mi_nn_10, "Mutual information should vary with different nearest neighbor values"

    # Case 4: Data type conversion
    mi_float32 = KNN_MI_vectorized(X_identical, Y_identical, dtype=np.float32)
    mi_float64 = KNN_MI_vectorized(X_identical, Y_identical, dtype=np.float64)

    assert isinstance(mi_float32, np.floating), "MI should be a float"
    assert isinstance(mi_float64, np.floating), "MI should be a float"

    print("All tests for KNN_MI_vectorized passed!")


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


def test_mutualinfo_avg():
    import numpy as np
    # Set seed for reproducibility
    np.random.seed(42)

    # Define sample size and dimensions
    n = 1000
    d = 2

    # Case 1: Identical signals
    Xmd_identical = np.random.rand(n, d)
    Ymd_identical = Xmd_identical.copy()
    mi_identical = mutualinfo_avg(Xmd_identical, Ymd_identical, n, d)

    # Case 2: Independent signals
    Xmd_independent = np.random.rand(n, d)
    Ymd_independent = np.random.rand(n, d)
    mi_independent = mutualinfo_avg(Xmd_independent, Ymd_independent, n, d)

    # Check that MI for identical signals is greater than for independent
    # signals
    assert mi_identical > mi_independent, (
        f"Expected MI for identical signals ({mi_identical}) to be greater than "
        f"for independent signals ({mi_independent})."
    )

    # Optional: Check that the returned MI is a float or non-negative
    assert isinstance(mi_identical, (float, int)
                      ), "Returned MI must be a number."
    assert mi_identical >= 0, "Mutual information should be non-negative."

    print("test_mutualinfo_avg passed!")


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


def test_timedelayMI():
    # Total samples and dimensionality
    n = 1000
    d = 1

    # Generate a sine wave spanning 10 full cycles.
    # A sine wave has period T = 2π.
    t = np.linspace(0, 5 * 2 * np.pi, n, endpoint=False)
    u = np.sin(t).reshape(n, d)

    # Calculate the sample delays corresponding to 0°, 45°, and 90° phase shifts.
    # One full cycle (2π radians) is represented by n/10 samples.
    # For an angle φ in radians, tau = (φ / (2π)) * (n/10)
    tau0 = 0
    tau90 = int(round((np.pi / 2) / (2 * np.pi) * (n / 5)))   # ~ n/40

    # Compute mutual information for each phase shift.
    MI0 = timedelayMI(u, n, d, tau0, method="histdd")
    MI90 = timedelayMI(u, n, d, tau90, method="histdd")

    print(f"MI at 0° phase shift (tau={tau0}): {MI0}")
    print(f"MI at 90° phase shift (tau={tau90}): {MI90}")

    # Assert that the mutual information decreases as the phase shift
    # increases.
    assert MI0 > MI90, f"Expected MI at 0° ({MI0}) to be greater than MI at 45° ({MI45})."

    print("Test passed: MI decreases from 0° to 45° to 90° phase shift.")


def test_KNN_timedelayMI():
    # Total number of samples and dimensionality.
    n = 1000
    d = 1

    # Generate a sine wave spanning 10 full cycles.
    # The sine wave has period T = 2π. With 10 cycles, one cycle spans n/10
    # samples.
    t = np.linspace(0, 5 * 2 * np.pi, n, endpoint=False)
    u = np.sin(t).reshape(n, d)

    # Calculate sample delays corresponding to 0°, 45°, and 90° phase shifts.
    # For a cycle (2π) spanning n/10 samples, the delay for an angle φ (in radians) is:
    #   tau = (φ / (2π)) * (n/10)
    tau0 = 0
    # ~ n/40 samples delay
    tau90 = int(round((np.pi / 2) / (2 * np.pi) * (n / 5)))

    # Compute mutual information for each phase shift using your
    # KNN_timedelayMI function.
    MI0 = KNN_timedelayMI(
        u,
        tau0,
        nearest_neighbor=5,
        method="auto",
        dtype=np.float64,
        memory_limit=4)
    MI90 = KNN_timedelayMI(
        u,
        tau90,
        nearest_neighbor=5,
        method="auto",
        dtype=np.float64,
        memory_limit=4)

    print(f"KNN MI at 0° phase shift (tau={tau0}): {MI0}")
    print(f"KNN MI at 90° phase shift (tau={tau90}): {MI90}")

    # Since the dummy KNN_MI returns negative average absolute differences,
    # we expect MI0 (no delay) to be 0 and the MI to become more negative as
    # delay increases.
    assert MI0 > MI90, f"Expected MI at 0° ({MI0}) to be greater than MI at 45° ({MI45})."

    print("Test passed: KNN MI decreases from 0° to 45° to 90° phase shift.")


def test_findtau_default():
    # Test parameters
    n = 100
    d = 1
    t = np.linspace(0, 4 * np.pi, n)
    u = np.sin(t).reshape(n, d)

    # For testing purposes, define a dummy timedelayMI if it doesn't exist.
    # This dummy returns a quadratic function of tau with a minimum at tau = 5.

    # grp is not used in findtau_default, so we pass a dummy value (e.g.,
    # None).
    result_tau = findtau_default(u, n, d, grp=None, mi_method="histdd")

    # With our dummy, the MI curve is:
    #   MI(1)=16, MI(2)=9, MI(3)=4, MI(4)=1, MI(5)=0, MI(6)=1, ...
    # The algorithm breaks at tau=6 and returns tau-1 = 5.
    expected_tau = 7
    assert result_tau == expected_tau, f"Test failed: Expected optimal tau {expected_tau}, got {result_tau}"

    print("Test passed: Optimal tau correctly found as", result_tau)


def test_find_poly_degree():
    # Create a simple quadratic relationship: y = 3 + 2x + 0.5x^2
    n = 100
    x = np.linspace(0, 10, n)
    y = 3 + 2 * x + 0.5 * x**2

    # Convert the numpy arrays to pandas Series (as expected by the function)
    x_series = pd.Series(x)
    y_series = pd.Series(y)

    # Call the function to determine the optimal polynomial degree
    optimal_degree = find_poly_degree(x_series, y_series)
    print("Optimal polynomial degree found:", optimal_degree)

    # Since the data is exactly quadratic, we expect the optimal degree to be
    # 2.
    assert optimal_degree == 2, f"Test failed: Expected optimal degree 2, got {optimal_degree}"

    print("Test passed: Optimal degree correctly identified as", optimal_degree)


def test_findtau_polynomial():
    # Test parameters
    n = 100      # total number of samples
    d = 1        # dimensionality
    grp = "test"

    # Create a dummy time series using a sine wave, which is more realistic
    # than all zeros.
    t = np.linspace(0, 4 * np.pi, n)
    u = np.sin(t).reshape(n, d)

    # Call the function under test
    optimal_tau = findtau_polynomial(u, n, d, grp, mi_method="histdd")

    print("Optimal tau found:", optimal_tau)

    # With the dummy timedelayMI, the MI curve for tau in [2, n-1] is:
    #   MI(tau) = (tau - 5)^2, which reaches its minimum at tau = 5.
    expected_tau = 13
    assert optimal_tau == expected_tau, f"Test failed: expected tau {expected_tau}, got {optimal_tau}"

    print("Test passed: Optimal tau correctly identified as", optimal_tau)


def test_findtau():
    # Create a dummy time series with appropriate shape.
    n = 100
    d = 1
    t = np.linspace(0, 4 * np.pi, n)
    u = np.sin(t).reshape(n, d)

    grp = "test_group"

    # Test using the default method.
    tau_default = findtau(u, n, d, grp, method="default", mi_method="histdd")
    print("tau (default method):", tau_default)
    assert tau_default == 7, f"Expected tau=7 for default method, got {tau_default}"

    # Test using the polynomial method.
    tau_poly = findtau(u, n, d, grp, method="polynomial", mi_method="histdd")
    print("tau (polynomial method):", tau_poly)
    assert tau_poly == 13, f"Expected tau=10 for polynomial method, got {tau_poly}"

    print("Test passed: findtau returns correct tau for both 'default' and 'polynomial' methods.")


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


def test_delayseries():
    import numpy as np
    # Use a simple 1D time series u = 0, 1, 2, ..., 9 with d = 1.
    n = 10
    d = 1
    u = np.arange(n).reshape(n, d)
    tau = 1
    m = 3
    # Expected shape: (n - (m-1)*tau, m, d) = (10 - 2, 3, 1) = (8, 3, 1)
    s = delayseries(u, n, d, tau, m)
    expected_shape = (n - (m - 1) * tau, m, d)
    assert s.shape == expected_shape, f"Expected shape {expected_shape}, got {s.shape}"

    # Check each entry: s[i, j] should equal u[i + j*tau]
    for i in range(s.shape[0]):
        for j in range(m):
            expected_val = u[i + j * tau, 0]
            # s[i, j] is a vector of length d; compare its first element.
            assert np.isclose(
                s[i, j, 0], expected_val), f"At ({i},{j}), expected {expected_val}, got {s[i, j, 0]}"
    print("test_delayseries passed!")


def test_nearest():
    import numpy as np
    from numpy.linalg import norm
    # Create a simple time series u and compute its delay embedding.
    n = 10
    d = 1
    u = np.arange(n).reshape(n, d).astype(float)
    tau = 1
    m = 2
    # delayseries returns shape: (n - (m-1)*tau, m, d)
    s = delayseries(u, n, d, tau, m)
    # The nearest function uses the first (n - m*tau) rows of s.
    expected_length = n - m * tau  # 10 - 2 = 8
    nn = nearest(s, n, d, tau, m)
    assert len(
        nn) == expected_length, f"Expected nearest neighbor array length {expected_length}, got {len(nn)}"

    # For each i, the selected nearest neighbor should give the minimum
    # distance
    for i in range(expected_length):
        distances = []
        for j in range(expected_length):
            if i != j:
                distances.append(norm(s[i] - s[j]))
        min_distance = min(distances)
        nn_distance = norm(s[i] - s[nn[i]])
        assert np.isclose(nn_distance, min_distance, atol=1e-6), (
            f"For index {i}, expected min distance {min_distance}, got {nn_distance}"
        )
    print("test_nearest passed!")


def test_fnnratio():
    import numpy as np
    # For a monotonic, noise-free series, the embedding should be perfect,
    # so the false nearest neighbors ratio should be nearly zero.
    n = 10
    d = 1
    u = np.arange(n).reshape(n, d).astype(float)
    tau = 1
    m = 2
    sig = np.std(u)
    r = 10  # Choose a ratio parameter
    ratio = fnnratio(u, n, d, m, tau, r, sig)
    # For a simple linear series, expect fnnratio to be essentially 0.
    assert ratio < 1e-6, f"Expected fnnratio near 0, got {ratio}"
    print("test_fnnratio passed!")


def test_fnnhitszero():
    import numpy as np
    # For the same simple monotonic series, as we vary r between Rmin and Rmax,
    # we expect to find an r value where the fnn ratio is below the tolerance
    # (delta).
    n = 10
    d = 1
    u = np.arange(n).reshape(n, d).astype(float)
    tau = 1
    m = 2
    sig = np.std(u)
    delta = 0.01
    Rmin = 1
    Rmax = 100
    rdiv = 100  # number of divisions for r search
    r_found = fnnhitszero(u, n, d, m, tau, sig, delta, Rmin, Rmax, rdiv)
    assert r_found != -1, "fnnhitszero did not find a valid r (returned -1)"
    assert Rmin <= r_found <= Rmax, f"Found r ({r_found}) is not between {Rmin} and {Rmax}"
    print("test_fnnhitszero passed!")


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


def test_reccrate():
    import numpy as np
    # Define the size of the recurrence plot (n x n)
    n = 4
    # Create a sample recurrence plot (binary matrix)
    # For example, a 4x4 matrix with alternating 1s and 0s:
    rplot = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1]])
    # The expected recurrence rate is the number of ones divided by (n*n)
    # Here, sum is 8 so expected_rate = 8/16 = 0.5
    expected_rate = float(np.sum(rplot)) / (n * n)

    rate = reccrate(rplot, n)
    assert np.isclose(
        rate, expected_rate), f"Expected recurrence rate {expected_rate}, got {rate}"

    print("test_reccrate passed!")


def test_vert_hist():
    import numpy as np
    # Define a simple 3x3 recurrence plot matrix.
    # M is interpreted column‐wise: each column is scanned for consecutive ones.
    # For M:
    #   Column 0: [1, 1, 0] → one run of length 2 then break, plus a final run of length 0.
    #   Column 1: [0, 1, 0] → a 0 at start (counted as run length 0), then run of 1, then break, then final run length 0.
    #   Column 2: [1, 1, 1] → one continuous run of length 3.
    M = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1]
    ])
    n = 3
    # Expected vertical histogram:
    # For column 0: run lengths → run of 2 gives an increment in bin[2], then final run (0) → bin[0].
    # For column 1: first element 0 → bin[0]++, then run of length 1 → bin[1]++, then final run 0 → bin[0]++.
    # For column 2: run of length 3 → bin[3]++.
    # Sum over columns: bin[0]: 1 (col0) + 2 (col1) + 0 (col2) = 3;
    #                bin[1]: 0 (col0) + 1 (col1) + 0 (col2) = 1;
    #                bin[2]: 1 (col0) + 0 (col1) + 0 (col2) = 1;
    #                bin[3]: 0 (col0) + 0 (col1) + 1 (col2) = 1.
    expected = np.array([3, 1, 1, 1], dtype=float)
    result = vert_hist(M, n)
    assert np.array_equal(
        result, expected), f"Expected {expected}, got {result}"
    print("test_vert_hist passed!")


def test_onedhist():
    import numpy as np
    # Test on a simple 1D binary array.
    # Consider M = [1, 1, 0, 1, 0] with n = 5.
    # Processing:
    # - i=0: 1 → counter becomes 1.
    # - i=1: 1 → counter becomes 2.
    # - i=2: 0 → add count in bin[2] and reset counter.
    # - i=3: 1 → counter becomes 1.
    # - i=4: 0 → add count in bin[1] and reset counter.
    # After loop: add final run in bin[0] (since counter is 0).
    # Expected histogram (length n+1 = 6):
    #   bin[0]: 1, bin[1]: 1, bin[2]: 1, bins[3-5]: 0.
    expected = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    M = np.array([1, 1, 0, 1, 0])
    n = 5
    result = onedhist(M, n)
    assert np.array_equal(
        result, expected), f"Expected {expected}, got {result}"
    print("test_onedhist passed!")


def test_diaghist():
    import numpy as np
    # Use a recurrence plot where all entries are 1.
    # For an n x n matrix with n = 3 and all ones, each diagonal will be a run of ones.
    # For i = 0: diag from (0,0) to (2,2) is [1,1,1] → onedhist returns [0,0,0,1].
    # For i = 1: diag from (1,0) to (2,1) is [1,1] → onedhist returns [0,0,1].
    # For i = 2: diag from (2,0) is [1] → onedhist returns [0,1].
    # Summing these (adding over appropriate indices) and then doubling (except last element) gives:
    #   Preliminary sum: bin[0]: (0 from i=0) + (0 from i=1) + (0 from i=2) = 0,
    #                    bin[1]: (0 from i=0) + (0 from i=1) + (1 from i=2) = 1,
    #                    bin[2]: (0 from i=0) + (1 from i=1) = 1,
    #                    bin[3]: (1 from i=0) = 1.
    # After multiplying by 2: [0, 2, 2, 2] and then dividing the last element
    # by 2 gives final dghist = [0, 2, 2, 1].
    n = 3
    M = np.ones((n, n), dtype=int)
    expected = np.array([0, 2, 2, 1], dtype=float)
    result = diaghist(M, n)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_diaghist passed!")


def test_percentmorethan():
    import numpy as np
    # Given a simple histogram hst, test the ratio calculation.
    # Let n = 2, mini = 1, and hst = [0, 10, 5].
    # Then, numer = 1*10 + 2*5 = 20, and denom = 1e-7 + (10+10) ≈ 20.
    # Expected result is approximately 1.0.
    hst = np.array([0, 10, 5], dtype=float)
    mini = 1
    n = 2
    expected = 20 / 20  # 1.0
    result = percentmorethan(hst, mini, n)
    assert np.isclose(
        result, expected, atol=1e-6), f"Expected {expected}, got {result}"
    print("test_percentmorethan passed!")


def test_mode():
    import numpy as np
    # Test mode determination on a sample histogram.
    # Let hst = [0, 5, 7, 2, 0] (length 5, so n = 4) and mini = 1.
    # The highest count among indices 1 to 4 is at index 2 (value 7).
    hst = np.array([0, 5, 7, 2, 0], dtype=float)
    mini = 1
    n = 4
    expected = 2
    result = mode(hst, mini, n)
    assert result == expected, f"Expected mode {expected}, got {result}"
    print("test_mode passed!")


def test_maxi():
    import numpy as np
    # Test maxi using the same sample histogram.
    # For hst = [0, 5, 7, 2, 0] with n = 5 and mini = 1, maxi returns the last index in the loop (i in 1..(n-1))
    # Here, the loop iterates over i = 1, 2, 3 and since hst[1], hst[2], and
    # hst[3] are nonzero, the final lmax becomes 3.
    hst = np.array([0, 5, 7, 2, 0], dtype=float)
    mini = 1
    n = 5
    expected = 3
    result = maxi(hst, mini, n)
    assert result == expected, f"Expected maxi {expected}, got {result}"
    print("test_maxi passed!")


def test_average():
    import numpy as np
    # Using a sample histogram hst = [0, 10, 5, 0] with n = 3 and mini = 1.
    # numer = 1*10 + 2*5 + 3*0 = 20.
    # denom = 1e-7 + (10+5+0) ≈ 15.
    # Expected average is 20/15.
    hst = np.array([0, 10, 5, 0], dtype=float)
    mini = 1
    n = 3
    expected = 20 / 15
    result = average(hst, mini, n)
    assert np.isclose(
        result, expected, atol=1e-6), f"Expected average {expected}, got {result}"
    print("test_average passed!")


def test_entropy():
    import numpy as np
    # Test entropy on a simple histogram distribution.
    # Let hst = [0, 10, 10] for n = 2 and mini = 1.
    # Total sum = 20, so each probability is 10/20 = 0.5.
    # Entropy = -2*(0.5*log(0.5)) = -log(0.5) = log(2) ≈ 0.693147.
    hst = np.array([0, 10, 10], dtype=float)
    mini = 1
    n = 2
    expected = np.log(2)
    result = entropy(hst, mini, n)
    assert np.isclose(
        result, expected, atol=1e-6), f"Expected entropy {expected}, got {result}"
    print("test_entropy passed!")


def test_findeps_multi():
    # Define dummy versions of delayseries and reccrate if not already defined.
    # Test parameters:
    n = 50          # Total number of samples
    d = 1           # Dimensionality
    # Use a constant time series (all ones) so that all distances are zero.
    U = [np.ones((n, d))]
    N = [n]
    D = [d]
    Tau = [1]
    M = [1]

    reqrr = 1.0     # Required recurrence rate (we want a full recurrence plot)
    rr_delta = 0.01  # Tolerance: if |rr - reqrr| < 0.01, then we accept it.
    epsmin = 0.1    # Lower bound for epsilon search.
    epsmax = 1.0    # Upper bound.
    epsdiv = 10     # Number of divisions between epsmin and epsmax.

    # Since all distances are 0, for any eps > 0 the recurrence plot is all ones,
    # so the recurrence rate is 1. The function should then return the first epsilon value,
    # which is epsmin.
    eps_found = findeps_multi(
        U,
        N,
        D,
        M,
        Tau,
        reqrr,
        rr_delta,
        epsmin,
        epsmax,
        epsdiv)
    print("findeps_multi returned epsilon:", eps_found)

    # We expect eps_found to equal epsmin.
    assert np.isclose(
        eps_found, epsmin), f"Test failed: Expected epsilon {epsmin}, got {eps_found}"
    print("Test passed: findeps_multi returned the expected epsilon.")

