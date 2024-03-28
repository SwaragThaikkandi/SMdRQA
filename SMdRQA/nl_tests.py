import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
from nolitsa import d2, dimension, surrogates
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from numpy.random import rand, randn
from datetime import datetime
from SMdRQA.RQA_functions import embedded_signal
from scipy import signal
import random
import numba


def wrapTo2Pi(x):
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


def Chi2_test(signal):
    '''
    Stationarity test for time series data

    The stationarity test proposed by Isliker & Kurths (1993)
    compares the distribution of the first half of the time series with
    the distribution of the entire time series using a chi-squared test.
    If there is no significant difference in the two distributions, we
    can consider the time series to be stationary.

    Parameters
    ----------
    signal : array
        One diensional time series signal

    Returns
    -------

    chisq : float
        Chi-square statistic value

    p : float
        p-vaue for chi-square test

    References
    ----------
    - Isliker, Heinz, and Juergen Kurths. "A test for stationarity: finding parts in time series apt for correlation dimension estimates.
      " International Journal of Bifurcation and Chaos 3.06 (1993): 1573-1579.
    - Mannattil, Manu, Himanshu Gupta, and Sagar Chakraborty. "Revisiting evidence of chaos in X-ray light curves: the case of GRS 1915+ 105."
      The Astrophysical Journal 833.2 (2016): 208.

    '''

    N = len(signal)
    reminder = N % 2
    if reminder == 0:
        Nhalf = int(N / 2)
        first_half = signal[0:Nhalf]

    elif reminder == 1:
        Nhalf = int(np.floor(N / 2) + 1)
        first_half = signal[0:Nhalf]

    nq, edges = np.histogram(signal)
    nqhalf, edges2 = np.histogram(first_half, bins=edges)
    Expected = ((Nhalf / N) * nq) + (10 ** (-9))
    (chisq, p) = ss.chisquare(f_obs=nqhalf, f_exp=Expected)

    return chisq, p


def Correlation_sum_plot(
    sig=None, r=100, metric="chebyshev", window=10, save_name=None
):
    '''
    Compute the correlation sum plots.

    Computes the correlation sum of the given time series for the specified distances (Grassberger & Procaccia 1983).

    Parameters
    ----------
    sig : ndarray
        N-dimensional real input array containing points in the phase space.
    r : int or array, optional (default = 100)
        Distances for which the correlation sum should be calculated.
        If r is an int, then the distances are taken to be a geometric progression between a minimum and maximum length scale
        (estimated according to the metric and the input series).
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "chebyshev" (aka the maximum norm metric), "cityblock" (aka the Manhattan metric), or "euclidean".
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist between pairs.
    save_path: str(optional)
        File name for saving the figure

    Returns
    -------
    figure

    References
    ----------
    - Grassberger, Peter, and Itamar Procaccia. "Characterization of strange attractors." Physical review letters 50.5 (1983): 346.
    - Mannattil, Manu, Himanshu Gupta, and Sagar Chakraborty. "Revisiting evidence of chaos in X-ray light curves: the case of GRS 1915+ 105." The Astrophysical Journal 833.2 (2016): 208.
    '''
    r, c = d2.c2(sig, r=r, metric=metric, window=window)
    plt.figure(figsize=(16, 12))
    plt.plot(r, c, "b-")
    plt.title("Correlation Sum Plot")
    plt.xlabel("r")
    plt.ylabel("C(r)")
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def time_irreversibility(sig):
    '''
    Compute the time irriversibility .

    Computes the time irriversibility of the signal

    Parameters
    ----------
    sig : ndarray
        N-dimensional real input array containing points in the phase
        space.

    Returns
    -------
    TI : float
       time irriversibility of the signal

    References
    ----------
    - Discrimination power of measures for nonlinearity in a time series. Physical Review E, 55 (5), 5443.


    '''
    before = sig[:-1]
    after = sig[1:]
    array = (after - before) ** 3
    TI = (1 / len(sig)) * np.sum(array)
    return TI


def sort_matlab(arr, dim=None):
    '''
    Sorts the elements of an array along a specified dimension in ascending order,
    similar to MATLAB's sort function.

    Parameters:
    arr : numpy.ndarray
        The input array to be sorted.

    dim : int, optional
        The dimension along which to sort the array. Default is None, which means
        the array is flattened before sorting.

    Returns:
    sorted_arr : numpy.ndarray
        The sorted array.

    sorted_indices : numpy.ndarray
        The indices that would sort the original array.
    '''
    if dim is not None:
        sorted_indices = np.argsort(arr, axis=dim)
        sorted_arr = np.take_along_axis(arr, sorted_indices, axis=dim)
    else:
        flattened_indices = np.argsort(arr, axis=None)
        sorted_arr = np.take_along_axis(arr, flattened_indices, axis=None)
        sorted_indices = np.unravel_index(flattened_indices, arr.shape)

    return sorted_arr, sorted_indices


def preprocessing(sig, fs):
    '''
    pre processing for surrogate data generation

    function is used to truncate the input signal based on minimizing the mismatch between consecutive points at
    the beginning and end of the signal, along with some initial processing steps like    mean subtraction and
    time vector generation. Adapted from Gemma, et al.,2018

    Parameters
    ----------
    sig : ndarray
        1-dimensional real input array containing points in the phase
        space.

    fs : int
       sampling rate of the signal

    Returns
    -------
    cutsig : ndarray
       1-dimensional array of truncated signal

    t2: float
       total length(in terms of time) of the truncated signal

    kstart : float
       starting point of the truncated signal w.r.t the original signal

    kend : float
       ending point of the truncated signal w.r.t the original signal

    References
    ----------
    - Lancaster, Gemma, et al. "Surrogate data for hypothesis testing of physical systems." Physics Reports 748 (2018): 1-60.


    '''
    sig = sig - np.mean(sig)
    t = np.linspace(0, len(sig) / fs, len(sig))
    L = len(sig)
    p = 10  # Find pair of points which minimizes mismatch between p consecutive
    # points and the beginning and the end of the signal
    K1 = round(L / 100)  # Proportion of signal to consider at the beginning
    k1 = sig[:K1]
    K2 = round(L / 10)  # Proportion of signal to consider at the end
    k2 = sig[-K2:]
    # Truncate to match start and end points and first derivatives
    if len(k1) <= p:
        p = len(k1) - 1
    d = np.zeros((len(k1) - p, len(k2) - p))
    for j in range(len(k1) - p):
        for k in range(len(k2) - p):
            d[j, k] = sum(abs(k1[j: j + p] - k2[k: k + p]))
    v, I = min(abs(d), axis=1)
    I2 = np.argmin(v)  # Minimum mismatch
    kstart = I2
    kend = I[I2] + len(sig[:-K2])
    if kend > len(sig):
        kend = len(sig)
    cutsig = sig[kstart:kend]  # New truncated time series
    t2 = t[kstart:kend]  # Corresponding time
    return cutsig, t2, kstart, kend


def surrogate(sig, N, method, pp, fs, *args):
    '''
    function for generating surrogate data

    function is used to generate surrogate data for hypothesis testing. Adapted from Gemma, et al.,2018

    Parameters
    ----------
    sig : ndarray
        1-dimensional real input array containing points in the phase
        space.

    N  : int
       Number of surrogates to calculate



    fs : int
       sampling rate of the signal

    pp : bool(default = True)
       preprocessing on (True) or off (False)

    method : str
       Method used for generating surrogates

       *RP* : Random permutation surrogates

       *FT* : Fourier transform surrogates

       *AAFT* : Amplitude adjusted Fourier transform

       *IAAFT1* : Iterative amplitude adjusted Fourier transform with exact distribution

       *IAAFT2* : Iterative amplitude adjusted Fourier transform with exact spectrum

       *CPP* : Cyclic phase permutation

       *PPS* : Pseudo-periodic

       *TS* : Twin

       *tshift* : Time shifted

       *CSS* : Cycle shuffled surrogates. Require that the signal can be
         separated into distinct cycles. May require adjustment of peak finding
         parameters.


    Returns
    -------
    surr : ndarray
       surrogate data

    params : dict
       parameters from the generation, including truncation locations if preprocessed and runtime.

    References
    ----------
    - Lancaster, Gemma, et al. "Surrogate data for hypothesis testing of physical systems." Physics Reports 748 (2018): 1-60.
    - Theiler, James, et al. "Testing for nonlinearity in time series: the method of surrogate data." Physica D: Nonlinear Phenomena 58.1-4 (1992): 77-94.
    - Schreiber, Thomas, and Andreas Schmitz. "Improved surrogate data for nonlinearity tests." Physical review letters 77.4 (1996): 635.
    - Small, Michael, Dejin Yu, and Robert G. Harrison. "Surrogate test for pseudoperiodic time series data." Physical Review Letters 87.18 (2001): 188101.
    - Thiel, Marco, et al. "Twin surrogates to test for complex synchronisation." Europhysics Letters 75.4 (2006): 535.
    - Theiler, James. "On the evidence for low-dimensional chaos in an epileptic electroencephalogram." Physics Letters A 196.1-2 (1994): 335-341.


    '''
    origsig = sig
    params = {}
    params["origsig"] = origsig
    params["method"] = method
    params["numsurr"] = N
    params["fs"] = fs
    z = datetime.now()
    # Preprocessing
    if pp:
        sig, time, ks, ke = preprocessing(sig, fs)
    else:
        time = np.linspace(0, len(sig) / fs, len(sig))
    L = len(sig)
    L2 = np.ceil(L / 2)
    if pp == 1:
        params["preprocessing"] = "on"
        params["cutsig"] = sig
        params["sigstart"] = ks
        params["sigend"] = ke
    else:
        params["preprocessing"] = "off"
    params["time"] = time
    # Random permutation (RP) surrogates
    if method == "RP":
        surr = np.zeros((N, len(sig)))
        for k in range(N):
            surr[k, :] = sig[np.random.permutation(L)]
    # Fourier transform (FT) surrogate
    elif method == "FT":
        a = 0
        b = 2 * np.pi
        if len(args) > 0:
            eta = args[0]
        else:
            eta = (b - a) * rand(N, L2 - 1) + a  # Random phases
        ftsig = fft(sig)  # Fourier transform of signal
        ftrp = np.zeros((N, len(ftsig)))
        ftrp[:, 0] = ftsig[0]
        F = ftsig[1:L2]
        F = F[np.ones(N), :]
        ftrp[:, 1:L2] = F * (np.exp(1j * eta))
        ftrp[:, 2 + L - L2: L] = np.conj(np.fliplr(ftrp[:, 2:L2]))
        surr = ifft(ftrp, axis=1)
        params["rphases"] = eta
    # Amplitude adjusted Fourier transform surrogate
    elif method == "AAFT":
        a = 0
        b = 2 * np.pi
        eta = (b - a) * np.random.randn(N, L2 - 1) + a  # Random phases
        val, ind = sort_matlab(sig)
        rankind = np.zeros(L)
        rankind[ind] = range(L)  # Rank the locations
        gn = sort_matlab(
            np.random.randn(N, len(sig)), dim=1
        )  # Create Gaussian noise signal and np.np.sort
        for j in range(N):
            gn[j, :] = gn[
                j, rankind
            ]  # Reorder noise signal to match ranks in original signal
        ftgn = fft(gn, axis=1)
        F = ftgn[:, 1:L2]
        surr = np.zeros((N, len(sig)))
        surr[:, 0] = gn[:, 0]
        surr[:, 1:L2] = F * np.exp(1j * eta)
        surr[:, 2 + L - L2: L] = np.conj(np.flipr(surr[:, 2:L2]))
        surr = ifft(surr, axis=1)
        _, ind2 = sort_matlab(surr, dim=1)  # Sort surrogate
        rrank = np.zeros(L)
        for k in range(N):
            rrank[ind2[k, :]] = range(L)
            surr[k, :] = val[rrank]
    # Iterated amplitude adjusted Fourier transform (IAAFT-1) with exact
    # distribution
    elif method == "IAAFT1":
        maxit = 1000
        val, ind = sort_matlab(sig)  # Sorted list of values
        rankind = np.zeros(L)  # Rank the values
        rankind[ind] = range(L)
        ftsig = fft(sig)
        F = ftsig[np.ones(N), :]
        surr = np.zeros((N, L))
        it = 1
        irank = rankind[np.ones(N), :]
        irank2 = np.zeros(L)
        oldrank = np.zeros((N, L))
        iind = np.zeros((N, L))
        iterf = np.zeros((N, L))
        while max(max(abs(oldrank - irank), axis=1)) != 0 and it < maxit:
            go = max(abs(oldrank - irank), axis=1)
            inc = np.where(go != 0)[0]
            oldrank = irank
            iterf[inc, :] = np.real(
                ifft(
                    abs(F[inc, :]) * np.exp(1j * np.angle(fft(surr[inc, :], axis=1))), axis=1
                )
            )
            _, iind[inc, :] = sort_matlab(iterf[inc, :], dim=1)
            for k in inc:
                irank2[iind[k, :]] = range(L)
                irank[k, :] = irank2
                surr[k, :] = val[irank2]
            it += 1
    # Iterated amplitude adjusted Fourier transform (IAAFT-2) with exact
    # spectrum
    elif method == "IAAFT2":
        maxit = 1000
        val, ind = sort_matlab(sig)  # Sorted list of values
        rankind = np.zeros(L)  # Rank the values
        rankind[ind] = range(L)
        ftsig = fft(sig)
        F = ftsig[np.ones(N), :]
        surr = np.zeros((N, L))
        it = 1
        irank = rankind[np.ones(N), :]
        irank2 = np.zeros(L)
        oldrank = np.zeros((N, L))
        iind = np.zeros((N, L))
        iterf = np.zeros((N, L))
        while max(max(abs(oldrank - irank), axis=1)) != 0 and it < maxit:
            go = max(abs(oldrank - irank), axis=1)
            inc = np.where(go != 0)[0]
            oldrank = irank
            iterf[inc, :] = np.real(
                ifft(
                    abs(F[inc, :]) * np.exp(1j * np.angle(fft(surr[inc, :], axis=1))), axis=1
                )
            )
            _, iind[inc, :] = sort_matlab(iterf[inc, :], dim=1)
            for k in inc:
                irank2[iind[k, :]] = range(L)
                irank[k, :] = irank2
                surr[k, :] = val[irank2]
            it += 1
        surr = iterf
    # Cyclic phase permutation (CPP) surrogates
    elif method == "CPP":
        phi = wrapTo2Pi(sig)
        pdiff = phi[1:] - phi[:-1]
        locs = np.where(pdiff < -np.pi)[0]
        parts = []
        for j in range(len(locs) - 1):
            tsig = phi[locs[j] + 1: locs[j + 1]]
            parts.append(tsig)
        st = phi[: locs[0]]
        en = phi[locs[-1] + 1:]
        surr = np.zeros((N, L))
        for k in range(N):
            surr[k, :] = np.unwrap(
                np.hstack((st, parts[np.random.permutation(len(parts))], en)))
    # Pseudo-periodic surrogates (PPS)
    elif method == "PPS":
        # Embedding of original signal
        sig, tau, m = embedded_signal(data=np.reshape(sig, (len(sig), 1)))
        L = len(sig)
        L2 = np.ceil(L / 2)
        time = np.linspace(0, len(sig) / fs, len(sig))
        params["embed_delay"] = tau
        params["embed_dim"] = m
        params["embed_sig"] = sig
        # Find the index of the first nearest neighbour from the first half of the
        # embedded signal to its last value to avoid cycling near last value
        matr = max(abs(sig[:, :] - sig[:, k] * np.ones(1, L))
                   for k in range(L))
        ssig, mind = min(matr[matr > 0], axis=1)
        _, pl = min(matr[: round(L / 2), L])
        rho = 0.7 * np.mean(ssig)
        for x in range(N):
            kn = random.randint(L)  # Choose random starting point
            for j in range(
                L
            ):  # Length of surrogate is the same as the embedded time series
                if kn == L:
                    kn = pl
                kn += 1  # Move forward from previous kn
                # Set surrogate to current value for kn (choose first
                # component, can be any)
                surr[x, j] = sig[0, kn]
                sigdist = max(
                    abs(sig[:, :] - sig[:, kn] * np.ones(1, L))
                )  # Find the maximum
                # distance between each point in the original signal and the current
                # values with noise added
                _, kn = min(sigdist)  # Find nearest point
    # Twin surrogates
    elif method == "TS":
        # Embedding of original signal
        sig, tau, m = embedded_signal(data=np.reshape(sig, (len(sig), 1)))
        L = len(sig)
        L2 = np.ceil(L / 2)
        time = np.linspace(0, len(sig) / fs, len(sig))
        params["embed_delay"] = tau
        params["embed_dim"] = m
        params["embed_sig"] = sig
        dL = L
        alpha = 0.1
        Rij = np.zeros((L, L))
        for k in range(1, L):
            Rij[k, :k] = max(abs(sig[:, :k] - sig[:, k] * np.ones(1, k)))
        Rij = Rij + Rij.T
        _, pl = min(Rij[: round(L / 2), L])
        Sij = sort_matlab(Rij.flatten())
        delta = Sij[round(alpha * L**2)]
        Rij[Rij < delta] = -1
        Rij[Rij > delta] = 0
        Rij = abs(Rij)
        ind = [[] for _ in range(L)]
        eln = np.zeros(L)
        twind = np.arange(L)
        remp = np.arange(1, L + 1)  # remaining points
        while len(remp) > 0:
            twn = remp[0]
            ind[twn] = remp[max(
                abs(Rij[:, remp] - Rij[:, twn] * np.ones(1, len(remp)))) == 0]
            ind[ind[twn]] = ind[twn]
            eln[ind[twn]] = len(ind[twn])
            twind[ind[twn]] = 0
            remp = twind[twind > 0]
        for sn in range(N):
            kn = random.randint(L) - 1
            for j in range(dL):
                kn += 1
                surr[sn, j] = sig[0, kn]
                kn = ind[kn][random.randint(eln[kn])]
                if kn == L:
                    kn = pl
    # Time-shifted surrogates
    elif method == "tshift":
        for sn in range(N):
            startp = random.randint(L - 1)
            surr[sn, :] = np.hstack((sig[1 + startp: L], sig[1:startp]))
    # Cycle shuffled surrogates
    elif method == "CSS":
        if len(args) > 0:
            MPH = args[0]  # Minimum heak height
            MPD = args[1]  # Minimum peak distance
        else:
            MPH = 0
            MPD = fs
        I = signal.find_peaks(sig, height=MPH, distance=MPD)
        st = sig[: I[0] - 1]
        en = sig[I[-1]:]
        parts = [sig[I[j]: I[j + 1] - 1] for j in range(len(I) - 1)]
        for k in range(N):
            surr[k, :] = np.unwrap(
                np.hstack((st, parts[np.random.permutation(len(parts))], en)))
    params["runtime"] = (datetime.now() - z).total_seconds()
    params["type"] = method
    return surr, params
