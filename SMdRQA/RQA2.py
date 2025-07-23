from __future__ import annotations

import pathlib
import numpy as np
import scipy.fft as _fft
import scipy.stats as stats
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from dataclasses import dataclass, field
from typing import Literal, Sequence, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import pickle
from scipy.spatial import distance
from scipy.special import digamma
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from scipy.stats import skew
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Tuple, Optional

class RQA2:
    """
    Comprehensive Recurrence Quantification Analysis class that handles all RQA computations,
    visualizations, and batch processing in an object-oriented manner.
    
    Fixed version with proper 0-based indexing throughout.
    """
    
    def __init__(self, data=None, normalize=True, **kwargs):
        """
        Initialize RQA2 object with time series data.
        
        Parameters
        ----------
        data : ndarray, optional
            Time series data of shape (n_samples, n_dimensions)
        normalize : bool, default=True
            Whether to normalize the data (z-score normalization)
        **kwargs : dict
            Additional parameters for RQA computation
        """
        # Data properties
        self.data = None
        self.original_data = None
        self.n_samples = 0
        self.n_dimensions = 0
        
        # Computed parameters
        self._tau = None
        self._m = None
        self._eps = None
        self._recurrence_plot = None
        self._embedded_signal = None
        self._rqa_measures = {}
        
        # Configuration parameters with defaults
        self.config = {
            'rdiv': kwargs.get('rdiv', 451),
            'Rmin': kwargs.get('Rmin', 1),
            'Rmax': kwargs.get('Rmax', 10),
            'delta': kwargs.get('delta', 0.001),
            'bound': kwargs.get('bound', 0.2),
            'reqrr': kwargs.get('reqrr', 0.1),
            'rr_delta': kwargs.get('rr_delta', 0.005),
            'epsmin': kwargs.get('epsmin', 0),
            'epsmax': kwargs.get('epsmax', 10),
            'epsdiv': kwargs.get('epsdiv', 1001),
            'mi_method': kwargs.get('mi_method', 'histdd'),
            'tau_method': kwargs.get('tau_method', 'default'),
            'lmin': kwargs.get('lmin', 2)
        }
        
        if data is not None:
            self.load_data(data, normalize)
    
    # Data handling methods
    def load_data(self, data, normalize=True):
        """Load and optionally normalize time series data."""
        self.original_data = np.array(data)
        if self.original_data.ndim == 1:
            self.original_data = self.original_data.reshape(-1, 1)
        
        self.n_samples, self.n_dimensions = self.original_data.shape
        
        if normalize:
            self.data = (self.original_data - np.mean(self.original_data, axis=0, keepdims=True)) / \
                       (np.std(self.original_data, axis=0, keepdims=True) + 1e-10)
        else:
            self.data = self.original_data.copy()
        
        # Reset computed values
        self._reset_computed_values()
    
    def _reset_computed_values(self):
        """Reset all computed values when new data is loaded."""
        self._tau = None
        self._m = None
        self._eps = None
        self._recurrence_plot = None
        self._embedded_signal = None
        self._rqa_measures = {}
    
    def _embedded_length(self, m, tau):
        """Number of valid delay-vectors for (m, τ) - FIXED indexing."""
        return max(0, self.n_samples - (m - 1) * tau)
    
    # Properties for computed values
    @property
    def tau(self):
        """Time delay parameter."""
        if self._tau is None:
            self._tau = self.compute_time_delay()
        return int(self._tau)
    
    @property
    def m(self):
        """Embedding dimension."""
        if self._m is None:
            self._m = self.compute_embedding_dimension()
        return int(self._m)
    
    @property
    def eps(self):
        """Neighborhood radius."""
        if self._eps is None:
            self._eps = self.compute_neighborhood_radius()
        return float(self._eps)
    
    @property
    def recurrence_plot(self):
        """Recurrence plot matrix."""
        if self._recurrence_plot is None:
            self._recurrence_plot = self.compute_recurrence_plot()
        return self._recurrence_plot
    
    @property
    def embedded_signal(self):
        """Time-delayed embedded signal."""
        if self._embedded_signal is None:
            self._embedded_signal = self.compute_embedded_signal()
        return self._embedded_signal
    
    @property
    def recurrence_rate(self):
        """Recurrence rate of the RP."""
        if self._recurrence_plot is not None:
            n = self._recurrence_plot.shape[0]
            return float(np.sum(self._recurrence_plot)) / (n * n)
        return None
    
    # Core computation methods
    def compute_time_delay(self, method=None, mi_method=None):
        """Compute optimal time delay using mutual information."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        method = method or self.config['tau_method']
        mi_method = mi_method or self.config['mi_method']
        
        if method == 'default':
            tau = self._findtau_default(mi_method)
        elif method == 'polynomial':
            tau = self._findtau_polynomial(mi_method)
        else:
            raise ValueError("Method must be 'default' or 'polynomial'")
        
        self._tau = int(tau)
        return self._tau
    
    def compute_embedding_dimension(self):
        """Compute optimal embedding dimension using false nearest neighbors."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        tau = self.tau
        sd = 3 * np.std(self.data)
        
        m = self._findm(tau, sd)
        self._m = int(m)
        return self._m
    
    def compute_neighborhood_radius(self, reqrr=None):
        """Compute neighborhood radius for specified recurrence rate."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        reqrr = reqrr or self.config['reqrr']
        reqrr = max(0.01, min(0.99, reqrr))
        tau = self.tau
        m = self.m
        
        eps = self._findeps(tau, m, reqrr)
        self._eps = float(eps)
        return self._eps
    
    def compute_recurrence_plot(self):
        """Compute the recurrence plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        tau = self.tau
        m = self.m
        eps = self.eps
        
        rplot = self._reccplot(tau, m, eps)
        self._recurrence_plot = rplot
        return rplot
    
    def compute_embedded_signal(self):
        """Compute time-delayed embedded signal."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        tau = self.tau
        m = self.m
        
        embedded = self._delayseries(tau, m)
        self._embedded_signal = embedded
        return embedded
    
    # RQA measures computation
    def compute_rqa_measures(self, lmin=None):
        """Compute all RQA measures."""
        lmin = lmin or self.config['lmin']
        rp = self.recurrence_plot
        n = rp.shape[0]
        
        # Compute line distributions
        diag_hist = self._diaghist(rp, n)
        vert_hist = self._vert_hist(rp, n)
        
        measures = {
            'recurrence_rate': self.recurrence_rate,
            'determinism': self._percentmorethan(diag_hist, lmin, n),
            'laminarity': self._percentmorethan(vert_hist, lmin, n),
            'diagonal_entropy': self._entropy(diag_hist, lmin, n),
            'vertical_entropy': self._entropy(vert_hist, lmin, n),
            'average_diagonal_length': self._average(diag_hist, lmin, n),
            'average_vertical_length': self._average(vert_hist, lmin, n),
            'max_diagonal_length': self._maxi(diag_hist, lmin, n),
            'max_vertical_length': self._maxi(vert_hist, lmin, n),
            'diagonal_mode': self._mode(diag_hist, lmin, n),
            'vertical_mode': self._mode(vert_hist, lmin, n)
        }
        
        self._rqa_measures = measures
        return measures
    
    def determinism(self, lmin=None):
        """Compute determinism (DET)."""
        if 'determinism' not in self._rqa_measures:
            self.compute_rqa_measures(lmin)
        return self._rqa_measures['determinism']
    
    def laminarity(self, lmin=None):
        """Compute laminarity (LAM)."""
        if 'laminarity' not in self._rqa_measures:
            self.compute_rqa_measures(lmin)
        return self._rqa_measures['laminarity']
    
    def trapping_time(self, lmin=None):
        """Compute trapping time (TT) - average vertical line length."""
        if 'average_vertical_length' not in self._rqa_measures:
            self.compute_rqa_measures(lmin)
        return self._rqa_measures['average_vertical_length']
    
    # Plotting methods
    def plot_recurrence_plot(self, figsize=(8, 8), title=None, save_path=None):
        """Plot the recurrence plot."""
        rp = self.recurrence_plot
        
        plt.figure(figsize=figsize)
        plt.imshow(rp, cmap='binary', origin='lower')
        plt.title(title or f'Recurrence Plot (RR={self.recurrence_rate:.3f})')
        plt.xlabel('Time Index')
        plt.ylabel('Time Index')
        plt.colorbar(label='Recurrence')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tau_mi_curve(self, max_tau=None, figsize=(10, 6), save_path=None):
        """Plot tau vs mutual information curve."""
        max_tau = max_tau or min(100, self.n_samples // 4)
        
        tau_values = []
        mi_values = []
        
        for tau in range(1, max_tau + 1):
            mi = self._timedelayMI(tau)
            tau_values.append(tau)
            mi_values.append(mi)
        
        plt.figure(figsize=figsize)
        plt.plot(tau_values, mi_values, 'b-o', markersize=4)
        plt.axvline(x=self.tau, color='r', linestyle='--', 
                   label=f'Optimal τ = {self.tau}')
        plt.xlabel('Time Delay (τ)')
        plt.ylabel('Mutual Information')
        plt.title('Time Delay vs Mutual Information')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fnn_curve(self, max_m=None, figsize=(10, 6), save_path=None):
        """Plot false nearest neighbors vs embedding dimension."""
        max_m = max_m or min(15, (3 * self.n_dimensions + 11) // 2)
        tau = self.tau
        sd = 3 * np.std(self.data)
        
        m_values = []
        fnn_values = []
        
        for m in range(1, max_m + 1):
            fnn = self._fnnratio(m, tau, 10, sd)
            m_values.append(m)
            fnn_values.append(fnn)
        
        plt.figure(figsize=figsize)
        plt.plot(m_values, fnn_values, 'g-o', markersize=4)
        plt.axvline(x=self.m, color='r', linestyle='--', 
                   label=f'Optimal m = {self.m}')
        plt.xlabel('Embedding Dimension (m)')
        plt.ylabel('False Nearest Neighbors Ratio')
        plt.title('Embedding Dimension vs False Nearest Neighbors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rqa_measures_summary(self, figsize=(12, 8), save_path=None):
        """Plot summary of RQA measures."""
        measures = self.compute_rqa_measures()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('RQA Measures Summary', fontsize=16)
        
        main_measures = ['recurrence_rate', 'determinism', 'laminarity']
        axes[0, 0].bar(main_measures, [measures[m] for m in main_measures])
        axes[0, 0].set_title('Main RQA Measures')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        entropy_measures = ['diagonal_entropy', 'vertical_entropy']
        axes[0, 1].bar(entropy_measures, [measures[m] for m in entropy_measures])
        axes[0, 1].set_title('Entropy Measures')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        avg_measures = ['average_diagonal_length', 'average_vertical_length']
        axes[0, 2].bar(avg_measures, [measures[m] for m in avg_measures])
        axes[0, 2].set_title('Average Line Lengths')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        max_measures = ['max_diagonal_length', 'max_vertical_length']
        axes[1, 0].bar(max_measures, [measures[m] for m in max_measures])
        axes[1, 0].set_title('Maximum Line Lengths')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        mode_measures = ['diagonal_mode', 'vertical_mode']
        axes[1, 1].bar(mode_measures, [measures[m] for m in mode_measures])
        axes[1, 1].set_title('Mode Line Lengths')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        params = ['tau', 'm', 'eps']
        param_values = [self.tau, self.m, self.eps]
        axes[1, 2].bar(params, param_values)
        axes[1, 2].set_title('RQA Parameters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series(self, figsize=(12, 6), save_path=None):
        """Plot the original time series data."""
        if self.original_data is None:
            raise ValueError("No data to plot.")
        
        plt.figure(figsize=figsize)
        
        if self.n_dimensions == 1:
            plt.plot(self.original_data.flatten())
            plt.title('Time Series')
            plt.xlabel('Time Index')
            plt.ylabel('Value')
        else:
            for i in range(min(self.n_dimensions, 5)):
                plt.subplot(min(self.n_dimensions, 5), 1, i+1)
                plt.plot(self.original_data[:, i])
                plt.title(f'Dimension {i+1}')
                plt.xlabel('Time Index')
                plt.ylabel('Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Batch processing methods
    @classmethod
    def batch_process(cls, input_path, output_path, group_level=False, 
                     group_level_estimates=None, **kwargs):
        """Process multiple time series files in batch."""
        os.makedirs(output_path, exist_ok=True)
        
        files = [f for f in os.listdir(input_path) if f.endswith('.npy')]
        
        results = []
        error_files = []
        
        # First pass: compute individual parameters
        rqa_objects = []
        for file in tqdm(files, desc="Processing files"):
            try:
                file_path = os.path.join(input_path, file)
                data = np.load(file_path)
                
                # Ensure 2D data
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                elif data.ndim == 0:
                    raise ValueError("Scalar data not supported")
                
                # Check minimum samples
                if data.shape[0] < 10:
                    raise ValueError(f"Too few samples: {data.shape[0]}")
                
                rqa = cls(data, **kwargs)
                
                # Compute basic parameters
                tau = rqa.compute_time_delay()
                m = rqa.compute_embedding_dimension()
                eps = rqa.compute_neighborhood_radius()
                
                rqa_objects.append(rqa)
                results.append({
                    'file': file,
                    'tau': tau,
                    'm': m,
                    'eps': eps
                })
                
            except Exception as e:
                error_files.append({'file': file, 'error': str(e)})
                print(f"Error processing {file}: {e}")
                continue
        
        # Group-level parameter estimation if requested
        if group_level and group_level_estimates and results:
            group_params = {}
            
            if 'tau' in group_level_estimates:
                group_params['tau'] = int(np.mean([r['tau'] for r in results]))
            if 'm' in group_level_estimates:
                group_params['m'] = int(np.mean([r['m'] for r in results]))
            if 'eps' in group_level_estimates:
                group_params['eps'] = cls._compute_group_epsilon(rqa_objects, **kwargs)
        
        # Second pass: compute RPs and RQA measures
        for i, rqa in enumerate(tqdm(rqa_objects, desc="Computing RPs")):
            try:
                file = results[i]['file']
                
                # Use group parameters if specified
                if group_level and group_level_estimates:
                    if 'tau' in group_level_estimates:
                        rqa._tau = group_params['tau']
                    if 'm' in group_level_estimates:
                        rqa._m = group_params['m']
                    if 'eps' in group_level_estimates:
                        rqa._eps = group_params['eps']
                
                # Compute RP and measures
                rp = rqa.compute_recurrence_plot()
                measures = rqa.compute_rqa_measures()
                
                # Save results
                np.save(os.path.join(output_path, file), rp)
                
                # Update results with measures
                results[i].update(measures)
                
            except Exception as e:
                error_files.append({'file': results[i]['file'], 'error': str(e)})
                print(f"Error computing RP for {results[i]['file']}: {e}")
        
        # Save summary files
        if results:
            pd.DataFrame(results).to_csv(os.path.join(output_path, 'rqa_results.csv'), index=False)
        if error_files:
            pd.DataFrame(error_files).to_csv(os.path.join(output_path, 'error_report.csv'), index=False)
        
        return results, error_files
    
    # Utility methods
    def save_results(self, filepath):
        """Save computed RQA results to file."""
        results = {
            'data': self.data,
            'tau': self._tau,
            'm': self._m,
            'eps': self._eps,
            'recurrence_plot': self._recurrence_plot,
            'rqa_measures': self._rqa_measures,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, filepath):
        """Load pre-computed RQA results from file."""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.data = results['data']
        self._tau = results['tau']
        self._m = results['m']
        self._eps = results['eps']
        self._recurrence_plot = results['recurrence_plot']
        self._rqa_measures = results['rqa_measures']
        self.config.update(results.get('config', {}))
        
        if self.data is not None:
            self.n_samples, self.n_dimensions = self.data.shape
    
    def get_summary(self):
        """Get summary of computed RQA parameters and measures."""
        summary = {
            'Data Info': {
                'Samples': self.n_samples,
                'Dimensions': self.n_dimensions
            },
            'Parameters': {
                'Time Delay (τ)': self.tau,
                'Embedding Dimension (m)': self.m,
                'Neighborhood Radius (ε)': self.eps,
                'Recurrence Rate': self.recurrence_rate
            }
        }
        
        if self._rqa_measures:
            summary['RQA Measures'] = self._rqa_measures
        
        return summary
    
    # Internal computation methods - ALL FIXED FOR 0-BASED INDEXING
    def _findtau_default(self, mi_method):
        """Find optimal time delay using first minima of MI curve."""
        max_tau = min(100, self.n_samples // 4)
        if max_tau < 2:
            return 1
            
        min_mi = self._timedelayMI(1, mi_method)
        
        for tau in range(2, max_tau):
            next_mi = self._timedelayMI(tau, mi_method)
            if next_mi > min_mi:
                return tau - 1
            min_mi = next_mi
        
        return max_tau - 1
    
    def _findtau_polynomial(self, mi_method):
        """Find optimal time delay using polynomial fit to MI curve."""
        max_tau = min(100, self.n_samples // 4)
        if max_tau < 3:
            return self._findtau_default(mi_method)
            
        tau_values = []
        mi_values = []
        
        for tau in range(1, max_tau):  # Fixed: 1 to max_tau-1
            mi = self._timedelayMI(tau, mi_method)
            tau_values.append(tau)
            mi_values.append(mi)
        
        if len(tau_values) < 3:
            return self._findtau_default(mi_method)
        
        tau_values = np.array(tau_values)
        mi_values = np.array(mi_values)
        
        degree = self._find_poly_degree(tau_values, mi_values)
        
        coefficients = np.polyfit(tau_values, mi_values, degree)
        polynomial = np.poly1d(coefficients)
        y_pred = polynomial(tau_values)
        
        tau_index = self._find_first_minima_or_global_minima_index(y_pred)
        return int(tau_values[tau_index])
    
    def _timedelayMI(self, tau, method='histdd'):
        """Compute time-delayed mutual information."""
        if tau >= self.n_samples:
            return 0.0
        
        # Fixed indexing: ensure we don't exceed bounds
        max_idx = self.n_samples - tau
        if max_idx <= 0:
            return 0.0
            
        X = self.data[:max_idx, :]  # 0 to max_idx-1
        Y = self.data[tau:tau+max_idx, :]  # tau to tau+max_idx-1
        
        return self._mutualinfo(X, Y, method)
    
    def _mutualinfo(self, X, Y, method='histdd'):
        """Compute mutual information between two time series."""
        n, d = X.shape
        
        if method == "histdd":
            return self._mutualinfo_histdd(X, Y, n, d)
        elif method == "avg":
            return self._mutualinfo_avg(X, Y, n, d)
        else:
            return self._mutualinfo_histdd(X, Y, n, d)
    
    def _mutualinfo_histdd(self, X, Y, n, d):
        """Mutual information using multidimensional histogram."""
        if n == 0:
            return 0.0
            
        points = np.concatenate((X, Y), axis=1)
        bins = min(15, max(3, int(np.cbrt(n))))
        
        try:
            p_xy = np.histogramdd(points, bins=bins)[0] + 1e-12
            p_x = np.histogramdd(X, bins=bins)[0] + 1e-12
            p_y = np.histogramdd(Y, bins=bins)[0] + 1e-12
            
            p_xy /= np.sum(p_xy)
            p_x /= np.sum(p_x)
            p_y /= np.sum(p_y)
            
            return np.sum(p_xy * np.log2(p_xy)) - np.sum(p_x * np.log2(p_x)) - np.sum(p_y * np.log2(p_y))
        except:
            return 0.0
    
    def _mutualinfo_avg(self, X, Y, n, d):
        """Average mutual information across dimensions."""
        mi = 0
        for i in range(d):
            X_i = X[:, i].reshape(-1, 1)
            Y_i = Y[:, i].reshape(-1, 1)
            mi += self._mutualinfo_histdd(X_i, Y_i, n, 1)
        return mi / d
    
    def _findm(self, tau, sd):
        """Find optimal embedding dimension using FNN method."""
        mmax = min(int((3 * self.n_dimensions + 11) / 2), 10)
        
        # Check if we have enough samples
        min_samples_needed = (mmax + 1) * tau  # Fixed: need m+1 for FNN
        if self.n_samples <= min_samples_needed:
            mmax = max(1, (self.n_samples - tau) // tau)
        
        if mmax < 1:
            return 1
            
        rm = self._fnnhitszero(mmax, tau, sd)
        if mmax > 1:
            rmp = self._fnnhitszero(mmax + 1, tau, sd)
            
            if rm != -1 and rmp != -1 and rm - rmp > self.config['bound']:
                return mmax + 1
        
        for m in range(1, mmax):
            rmp = rm
            rm = self._fnnhitszero(mmax - m, tau, sd)
            if rm != -1 and rmp != -1 and rm - rmp > self.config['bound']:
                return mmax + 1 - m
        
        return max(1, mmax)
    
    def _fnnhitszero(self, m, tau, sd):
        """Find r value where FNN ratio hits zero."""
        # Fixed: Check proper bounds for embedding
        min_samples_needed = (m + 1) * tau
        if self.n_samples <= min_samples_needed:
            return -1
            
        r_values = np.linspace(self.config['Rmin'], self.config['Rmax'], self.config['rdiv'])
        
        for r in r_values:
            if self._fnnratio(m, tau, r, sd) < self.config['delta']:
                return r
        return -1
    
    def _fnnratio(self, m, tau, r, sd):
        """Compute false nearest neighbors ratio - FIXED INDEXING."""
        # Fixed: Check bounds for both m and m+1 embeddings
        min_samples_m = (m - 1) * tau + 1
        min_samples_mp1 = m * tau + 1
        
        if self.n_samples <= min_samples_mp1:
            return 1.0
        
        try:
            s1 = self._delayseries(tau, m)
            s2 = self._delayseries(tau, m + 1)
        except:
            return 1.0
        
        nn = self._nearest(s1)
        
        n_embedded = s1.shape[0]
        n_embedded_mp1 = s2.shape[0]
        
        # Fixed: Use minimum length to avoid indexing errors
        max_valid = min(n_embedded, n_embedded_mp1, len(nn))
        
        isneigh = np.zeros(max_valid)
        isfalse = np.zeros(max_valid)
        
        for i in range(max_valid):
            if nn[i] < max_valid:  # Fixed: ensure valid index
                disto = np.linalg.norm(s1[i] - s1[nn[i]]) + 1e-12
                distp = np.linalg.norm(s2[i] - s2[nn[i]])
                
                if disto < sd / r:
                    isneigh[i] = 1
                    if distp / disto > r:
                        isfalse[i] = 1
        
        return np.sum(isneigh * isfalse) / (np.sum(isneigh) + 1e-12)
    
    def _delayseries(self, tau, m):
        """Create time-delayed embedding - COMPLETELY FIXED INDEXING."""
        n_embedded = self._embedded_length(m, tau)
        
        if n_embedded <= 0:
            raise ValueError(f"Insufficient data for embedding: need {(m-1)*tau + 1} samples, have {self.n_samples}")
        
        s = np.zeros((n_embedded, m, self.n_dimensions))
        
        # Fixed: Proper 0-based indexing
        for j in range(m):
            start_idx = j * tau
            end_idx = start_idx + n_embedded
            if end_idx <= self.n_samples:  # Fixed: ensure we don't exceed bounds
                s[:, j, :] = self.data[start_idx:end_idx, :]
            else:
                raise ValueError(f"Index out of bounds in embedding: trying to access {end_idx} with array size {self.n_samples}")
        
        return s
    
    def _nearest(self, s):
        """Find nearest neighbors - FIXED INDEXING."""
        n_embedded = s.shape[0]
        if n_embedded == 0:
            return np.array([])
            
        nn = np.zeros(n_embedded, dtype=int)
        
        for i in range(n_embedded):
            # Fixed: Vectorized distance computation with proper bounds
            distances = np.linalg.norm(s[i:i+1] - s, axis=(1, 2)).flatten()
            distances[i] = np.inf  # Exclude self-match
            
            # Fixed: Ensure valid index
            nearest_idx = np.argmin(distances)
            nn[i] = int(nearest_idx)
        
        return nn
    
    def _findeps(self, tau, m, reqrr):
        """Find neighborhood radius - FIXED INDEXING."""
        eps_values = np.linspace(self.config['epsmin'], self.config['epsmax'], self.config['epsdiv'])
        
        if np.all(eps_values == 0):
            eps_values = np.linspace(0.001, 1.0, self.config['epsdiv'])
        
        n_embedded = self._embedded_length(m, tau)
        if n_embedded <= 0:
            return 0.1
        
        try:
            s = self._delayseries(tau, m)
        except:
            return 0.1
        
        s_flat = s.reshape(n_embedded, -1)
        
        for eps in eps_values:
            if eps <= 0:
                continue
                
            try:
                D = distance.cdist(s_flat, s_flat)
                rplot = (D < eps).astype(int)
                
                rr = float(np.sum(rplot)) / (n_embedded * n_embedded)
                
                if abs(rr - reqrr) < self.config['rr_delta']:
                    return eps
            except:
                continue
        
        return (self.config['epsmin'] + self.config['epsmax']) / 2
    
    def _reccplot(self, tau, m, eps):
        """Compute recurrence plot - FIXED INDEXING."""
        try:
            s = self._delayseries(tau, m)
        except ValueError as e:
            raise ValueError(f"Cannot compute recurrence plot: {e}")
            
        n_embedded = s.shape[0]
        
        if n_embedded == 0:
            return np.array([[]])
        
        # Fixed: Proper shape handling
        s_flat = s.reshape(n_embedded, -1)
        D = distance.cdist(s_flat, s_flat)
        
        rplot = (D < eps).astype(int)
        return rplot
    
    # RQA measure computation methods - FIXED INDEXING
    def _vert_hist(self, rplot, n):
        """Compute vertical line distribution."""
        if n == 0:
            return np.array([0])
            
        nvert = np.zeros(n + 1)
        
        for i in range(n):  # Fixed: 0 to n-1
            counter = 0
            for j in range(n):  # Fixed: 0 to n-1
                if j < rplot.shape[0] and i < rplot.shape[1]:  # Fixed: bounds check
                    if rplot[j, i] == 1:
                        counter += 1
                    else:
                        if counter < len(nvert):
                            nvert[counter] += 1
                        counter = 0
            if counter < len(nvert):
                nvert[counter] += 1
        
        return nvert
    
    def _diaghist(self, rplot, n):
        """Compute diagonal line distribution."""
        if n == 0:
            return np.array([0])
            
        dghist = np.zeros(n + 1)
        
        for i in range(n):  # Fixed: 0 to n-1
            diag_len = n - i
            if diag_len > 0:
                diag = np.zeros(diag_len)
                for j in range(diag_len):  # Fixed: 0 to diag_len-1
                    if (i + j) < rplot.shape[0] and j < rplot.shape[1]:
                        diag[j] = rplot[i + j, j]
                
                subdiaghist = self._onedhist(diag, diag_len)
                for k in range(min(len(subdiaghist), len(dghist))):
                    dghist[k] += subdiaghist[k]
        
        dghist *= 2
        if len(dghist) > n:
            dghist[n] /= 2
        
        return dghist
    
    def _onedhist(self, arr, n):
        """Compute 1D histogram of line lengths."""
        if n == 0 or len(arr) == 0:
            return np.array([1])
            
        hst = np.zeros(n + 1)
        counter = 0
        
        for i in range(len(arr)):  # Fixed: 0 to len(arr)-1
            if arr[i] == 1:
                counter += 1
            else:
                if counter < len(hst):
                    hst[counter] += 1
                counter = 0
        
        if counter < len(hst):
            hst[counter] += 1
            
        return hst
    
    def _percentmorethan(self, hst, mini, n):
        """Compute percentage of recurrent points in lines longer than mini."""
        if len(hst) == 0 or n == 0:
            return 0.0
            
        max_idx = min(len(hst), n + 1)
        numer = sum(i * hst[i] for i in range(mini, max_idx))
        denom = sum(i * hst[i] for i in range(1, max_idx)) + 1e-12
        return numer / denom
    
    def _average(self, hst, mini, n):
        """Compute average line length."""
        if len(hst) == 0 or n == 0:
            return 0.0
            
        max_idx = min(len(hst), n + 1)
        numer = sum(i * hst[i] for i in range(mini, max_idx))
        denom = sum(hst[i] for i in range(mini, max_idx)) + 1e-12
        return numer / denom
    
    def _entropy(self, hst, mini, n):
        """Compute entropy of line length distribution."""
        if len(hst) == 0 or n == 0:
            return 0.0
            
        max_idx = min(len(hst), n + 1)
        total = sum(hst[i] for i in range(mini, max_idx))
        if total == 0:
            return 0
        
        entropy = 0
        for i in range(mini, max_idx):
            if hst[i] > 0:
                p = hst[i] / total
                entropy -= p * np.log(p)
        
        return entropy
    
    def _mode(self, hst, mini, n):
        """Find mode of line length distribution."""
        if len(hst) == 0 or n == 0:
            return mini
            
        max_idx = min(len(hst), n + 1)
        mode_val = mini
        for i in range(mini + 1, max_idx):
            if hst[i] > hst[mode_val]:
                mode_val = i
        return mode_val
    
    def _maxi(self, hst, mini, n):
        """Find maximum line length."""
        if len(hst) == 0 or n == 0:
            return 1
            
        max_idx = min(len(hst), n + 1)
        for i in range(max_idx - 1, 0, -1):  # Fixed: max_idx-1 to 1
            if hst[i] > 0:
                return i
        return 1
    
    # Helper methods - FIXED INDEXING
    def _find_first_minima_or_global_minima_index(self, arr):
        """Find first local minimum or global minimum."""
        if len(arr) == 0:
            return None
        
        n = len(arr)
        if n == 1:
            return 0
        if n == 2:
            return 0 if arr[0] <= arr[1] else 1
            
        # Check first element
        if arr[0] < arr[1]:
            return 0
        
        # Check middle elements
        for i in range(1, n - 1):  # Fixed: 1 to n-2
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                return int(i)
        
        # Check last element
        if arr[n-1] < arr[n-2]:
            return n-1
        
        # Fallback to global minimum
        return int(np.argmin(arr))
    
    def _find_poly_degree(self, x, y):
        """Find optimal polynomial degree using cross-validation."""
        if len(x) < 3:
            return 1
            
        max_deg = min(len(x) - 1, 10)
        best_rmse = float('inf')
        best_degree = 1
        
        # Convert to pandas Series if needed
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        for deg in range(1, max_deg + 1):
            try:
                n_splits = min(5, len(x))
                if n_splits < 2:
                    break
                    
                cv = RepeatedKFold(n_splits=n_splits, n_repeats=3, random_state=1)
                mse_scores = []
                
                x_vals = x.values
                y_vals = y.values
                
                for train_idx, test_idx in cv.split(x_vals, y_vals):
                    # Fixed: ensure indices are within bounds
                    train_idx = train_idx[train_idx < len(x_vals)]
                    test_idx = test_idx[test_idx < len(x_vals)]
                    
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        continue
                        
                    x_train, x_test = x_vals[train_idx], x_vals[test_idx]
                    y_train, y_test = y_vals[train_idx], y_vals[test_idx]
                    
                    coefficients = np.polyfit(x_train, y_train, deg)
                    polynomial = np.poly1d(coefficients)
                    y_pred = polynomial(x_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    mse_scores.append(mse)
                
                if mse_scores:
                    rmse = np.sqrt(np.mean(mse_scores))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_degree = deg
                        
            except:
                continue
        
        return best_degree
    
    @staticmethod
    def _compute_group_epsilon(rqa_objects, **kwargs):
        """Compute group-level epsilon for multiple time series."""
        eps_values = [rqa.eps for rqa in rqa_objects if rqa._eps is not None and rqa._eps > 0]
        return float(np.mean(eps_values)) if eps_values else 0.1





@dataclass
class RQA2_simulators:
    """
    Simulate various chaotic dynamical systems for testing RQA2 and surrogate methods.
    Includes multiple well-known chaotic attractors with configurable parameters.
    """
    
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
    
    def rossler(self, 
                tmax: int = 10000,
                n: int = 2000, 
                Xi: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                a: float = 0.2,
                b: float = 0.2, 
                c: float = 5.7,
                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the Rössler attractor.
        
        For b=0.2, c=5.7:
        - a < 0.2: chaotic regime
        - a > 0.2: synchronous/periodic regime
        """
        def rossler_system(t, state):
            x, y, z = state
            dxdt = -y - z
            dydt = x + a * y
            dzdt = b + z * (x - c)
            return [dxdt, dydt, dzdt]
        
        t_span = (0, tmax * dt)
        t_eval = np.linspace(0, tmax * dt, tmax)
        
        sol = solve_ivp(rossler_system, t_span, Xi, t_eval=t_eval, 
                       method='RK45', rtol=1e-9, atol=1e-12)
        
        # Subsample to get n points
        step = len(sol.y[0]) // n
        indices = np.arange(0, len(sol.y[0]), step)[:n]
        
        return sol.y[0][indices], sol.y[1][indices], sol.y[2][indices]
    
    def lorenz(self,
               tmax: int = 10000,
               n: int = 2000,
               Xi: Tuple[float, float, float] = (1.0, 1.0, 1.0),
               sigma: float = 10.0,
               rho: float = 28.0,
               beta: float = 8.0/3.0,
               dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the Lorenz attractor."""
        def lorenz_system(t, state):
            x, y, z = state
            dxdt = sigma * (y - x)
            dydt = rho * x - y - x * z
            dzdt = x * y - beta * z
            return [dxdt, dydt, dzdt]
        
        t_span = (0, tmax * dt)
        t_eval = np.linspace(0, tmax * dt, tmax)
        
        sol = solve_ivp(lorenz_system, t_span, Xi, t_eval=t_eval, 
                       method='RK45', rtol=1e-9, atol=1e-12)
        
        step = len(sol.y[0]) // n
        indices = np.arange(0, len(sol.y[0]), step)[:n]
        
        return sol.y[0][indices], sol.y[1][indices], sol.y[2][indices]
    
    def henon(self,
              n: int = 2000,
              Xi: Tuple[float, float] = (0.1, 0.1),
              a: float = 1.4,
              b: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the Hénon map."""
        x, y = Xi
        X, Y = [x], [y]
        
        for i in range(n - 1):
            x_next = 1 - a * x**2 + y
            y_next = b * x
            X.append(x_next)
            Y.append(y_next)
            x, y = x_next, y_next
        
        return np.array(X), np.array(Y)
    
    def chua(self,
             tmax: int = 10000,
             n: int = 2000,
             Xi: Tuple[float, float, float] = (0.1, 0.1, 0.1),
             alpha: float = 15.6,
             beta: float = 28.0,
             m0: float = -1.143,
             m1: float = -0.714,
             dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate Chua's circuit attractor."""
        def chua_system(t, state):
            x, y, z = state
            # Piecewise-linear function
            if x >= 1:
                h = m1 * x + (m0 - m1)
            elif x >= -1:
                h = m0 * x
            else:
                h = m1 * x - (m0 - m1)
            
            dxdt = alpha * (y - x - h)
            dydt = x - y + z
            dzdt = -beta * y
            return [dxdt, dydt, dzdt]
        
        t_span = (0, tmax * dt)
        t_eval = np.linspace(0, tmax * dt, tmax)
        
        sol = solve_ivp(chua_system, t_span, Xi, t_eval=t_eval, 
                       method='RK45', rtol=1e-9, atol=1e-12)
        
        step = len(sol.y[0]) // n
        indices = np.arange(0, len(sol.y[0]), step)[:n]
        
        return sol.y[0][indices], sol.y[1][indices], sol.y[2][indices]
    
    def generate_test_battery(self) -> dict:
        """Generate a complete battery of test systems."""
        systems = {}
        
        # Rössler chaotic regime
        x, y, z = self.rossler(tmax=5000, n=2000, a=0.1)
        systems['rossler_chaotic'] = {'x': x, 'y': y, 'z': z, 'regime': 'chaotic'}
        
        # Rössler synchronous regime  
        x, y, z = self.rossler(tmax=5000, n=2000, a=0.3)
        systems['rossler_sync'] = {'x': x, 'y': y, 'z': z, 'regime': 'synchronous'}
        
        # Lorenz system
        x, y, z = self.lorenz(tmax=5000, n=2000)
        systems['lorenz'] = {'x': x, 'y': y, 'z': z, 'regime': 'chaotic'}
        
        # Hénon map
        x, y = self.henon(n=2000)
        systems['henon'] = {'x': x, 'y': y, 'regime': 'chaotic'}
        
        # Chua circuit
        x, y, z = self.chua(tmax=5000, n=2000)
        systems['chua'] = {'x': x, 'y': y, 'z': z, 'regime': 'chaotic'}
        
        return systems




Algorithm = Literal[
    "FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"
]

@dataclass
class RQA2_tests:
    """
    Generate surrogate data with comprehensive statistical validation
    against chaotic dynamical systems.
    """

    signal: np.ndarray
    seed: int | None = None
    max_workers: int = 1
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        if not np.issubdtype(self.signal.dtype, np.floating):
            raise TypeError("Input signal must be floating-point")

    # ------------------------------------------------------------------
    # Public façade with enhanced validation
    # ------------------------------------------------------------------
    def generate(
        self,
        kind: Algorithm = "FT",
        *,
        n_surrogates: int = 200,
        **kwargs,
    ) -> np.ndarray:
        """Generate surrogate ensemble with statistical validation."""
        _dispatcher = {
            "FT": self._ft,
            "AAFT": self._aaft,
            "IAAFT": self._iaaft,
            "IDFS": self._idfs,
            "WIAAFT": self._wiaaft,
            "PPS": self._pps,
        }
        if kind not in _dispatcher:
            raise KeyError(f"Unknown surrogate type {kind}")
        
        generator_fn = _dispatcher[kind]
        
        # Parallel generation for large ensembles
        if n_surrogates >= 50 and self.max_workers > 1:
            return self._parallel_generate(generator_fn, n_surrogates, **kwargs)
        else:
            return np.vstack(
                [generator_fn(**kwargs) for _ in range(n_surrogates)]
            )
    
    def _parallel_generate(self, generator_fn, n_surrogates: int, **kwargs) -> np.ndarray:
        """Parallel surrogate generation for computational efficiency."""
        surrogates = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(generator_fn, **kwargs) 
                for _ in range(n_surrogates)
            ]
            for future in as_completed(futures):
                surrogates.append(future.result())
        return np.vstack(surrogates)

    def comprehensive_validation(
        self,
        systems_data: Dict[str, Dict[str, np.ndarray]],
        n_surrogates: int = 200,
        save_path: str = "surrogate_validation_results.png"
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive validation framework testing all surrogate methods
        against multiple chaotic systems using various nonlinear metrics.
        """
        surrogate_methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
        metrics = [
            "lyapunov_exponent", "time_irreversibility", "sample_entropy",
            "correlation_dimension", "nonlinearity_index", "predictability"
        ]
        
        results = {}
        
        for system_name, system_data in systems_data.items():
            print(f"\nProcessing {system_name}...")
            system_results = {}
            
            # Use first component (x-coordinate) for analysis
            signal = system_data['x']
            self.signal = signal
            
            # Calculate metrics for original signal
            original_metrics = self._calculate_all_metrics(signal)
            
            for surrogate_method in surrogate_methods:
                print(f"  Testing {surrogate_method} surrogates...")
                method_results = {}
                
                # Generate surrogate ensemble
                surrogates = self.generate(surrogate_method, n_surrogates=n_surrogates)
                
                # Calculate metrics for all surrogates
                surrogate_metrics = {metric: [] for metric in metrics}
                for i in range(n_surrogates):
                    surrogate_signal = surrogates[i]
                    metrics_values = self._calculate_all_metrics(surrogate_signal)
                    for metric in metrics:
                        surrogate_metrics[metric].append(metrics_values[metric])
                
                # Calculate p-values
                for metric in metrics:
                    original_value = original_metrics[metric]
                    surrogate_values = np.array(surrogate_metrics[metric])
                    
                    # Two-sided test: probability that observed differs from surrogate
                    if not np.isnan(original_value) and not np.any(np.isnan(surrogate_values)):
                        # Rank-based p-value calculation
                        n_extreme = np.sum(
                            (surrogate_values <= original_value) | 
                            (surrogate_values >= original_value)
                        )
                        p_value = min(n_extreme, n_surrogates - n_extreme) / n_surrogates
                        p_value = 2 * p_value  # Two-sided test
                        p_value = min(p_value, 1.0)
                    else:
                        p_value = np.nan
                    
                    method_results[metric] = p_value
                
                system_results[surrogate_method] = method_results
            
            results[system_name] = system_results
        
        # Create visualization
        self._create_validation_heatmap(results, save_path)
        
        return results

    def _calculate_all_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive set of nonlinear dynamics metrics."""
        metrics = {}
        
        try:
            metrics["lyapunov_exponent"] = self._lyapunov_exponent(signal)
        except:
            metrics["lyapunov_exponent"] = np.nan
            
        try:
            metrics["time_irreversibility"] = self._time_irreversibility(signal)
        except:
            metrics["time_irreversibility"] = np.nan
            
        try:
            metrics["sample_entropy"] = self._sample_entropy(signal)
        except:
            metrics["sample_entropy"] = np.nan
            
        try:
            metrics["correlation_dimension"] = self._correlation_dimension(signal)
        except:
            metrics["correlation_dimension"] = np.nan
            
        try:
            metrics["nonlinearity_index"] = self._nonlinearity_index(signal)
        except:
            metrics["nonlinearity_index"] = np.nan
            
        try:
            metrics["predictability"] = self._predictability_index(signal)
        except:
            metrics["predictability"] = np.nan
        
        return metrics

    def _lyapunov_exponent(self, signal: np.ndarray, dim: int = 3, tau: int = 1) -> float:
        """Calculate largest Lyapunov exponent using Rosenstein method."""
        # Phase space reconstruction
        N = len(signal)
        M = N - (dim - 1) * tau
        
        if M < 50:  # Not enough points
            return np.nan
            
        embedded = np.zeros((M, dim))
        for i in range(dim):
            embedded[:, i] = signal[i * tau:i * tau + M]
        
        # Find nearest neighbors
        lyap_values = []
        
        for i in range(min(M - 1, 500)):  # Limit for computational efficiency
            distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))
            distances[i] = np.inf  # Exclude self
            
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] > 0:
                # Track divergence
                divergences = []
                for k in range(1, min(20, M - max(i, nearest_idx))):
                    if i + k < M and nearest_idx + k < M:
                        dist_k = np.sqrt(np.sum((embedded[i + k] - embedded[nearest_idx + k])**2))
                        if dist_k > 0:
                            divergences.append(np.log(dist_k / distances[nearest_idx]) / k)
                
                if divergences:
                    lyap_values.extend(divergences)
        
        return np.mean(lyap_values) if lyap_values else np.nan

    def _time_irreversibility(self, signal: np.ndarray) -> float:
        """Calculate time irreversibility using third-order statistics."""
        n = len(signal)
        if n < 10:
            return np.nan
            
        # Centered signal
        signal_centered = signal - np.mean(signal)
        
        # Calculate asymmetry measure
        irreversibility = 0
        count = 0
        
        for lag in range(1, min(n//10, 20)):
            if n - lag > 0:
                forward = signal_centered[lag:] * (signal_centered[:-lag]**2)
                backward = signal_centered[:-lag] * (signal_centered[lag:]**2)
                
                diff = np.mean(forward) - np.mean(backward)
                irreversibility += diff**2
                count += 1
        
        return np.sqrt(irreversibility / count) if count > 0 else np.nan

    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy."""
        N = len(signal)
        if r is None:
            r = 0.2 * np.std(signal)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j]) <= r:
                        C[i] += 1.0
            
            phi = np.mean([np.log(c / (N - m + 1.0)) for c in C if c > 0])
            return phi
        
        return _phi(m) - _phi(m + 1)

    def _correlation_dimension(self, signal: np.ndarray, dim: int = 5) -> float:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
        # Phase space reconstruction
        N = len(signal)
        if N < 100:
            return np.nan
            
        tau = self._estimate_delay(signal)
        M = N - (dim - 1) * tau
        
        if M < 50:
            return np.nan
            
        embedded = np.zeros((M, dim))
        for i in range(dim):
            embedded[:, i] = signal[i * tau:i * tau + M]
        
        # Calculate correlation integral for different radii
        radii = np.logspace(-2, 0, 20) * np.std(signal)
        correlations = []
        
        for r in radii:
            count = 0
            total_pairs = 0
            
            # Sample pairs for computational efficiency
            for i in range(0, min(M, 200), 5):
                for j in range(i + 1, min(M, 200), 5):
                    dist = np.sqrt(np.sum((embedded[i] - embedded[j])**2))
                    if dist < r:
                        count += 1
                    total_pairs += 1
            
            if total_pairs > 0:
                correlations.append(count / total_pairs)
            else:
                correlations.append(0)
        
        # Fit linear region
        log_radii = np.log(radii)
        log_correlations = np.log(np.maximum(correlations, 1e-10))
        
        valid_idx = np.isfinite(log_correlations)
        if np.sum(valid_idx) < 5:
            return np.nan
            
        slope, _, _, _, _ = stats.linregress(
            log_radii[valid_idx], log_correlations[valid_idx]
        )
        
        return slope

    def _nonlinearity_index(self, signal: np.ndarray) -> float:
        """Calculate nonlinearity index based on phase space asymmetry."""
        # Simple nonlinearity measure based on skewness of increments
        increments = np.diff(signal)
        return abs(stats.skew(increments))

    def _predictability_index(self, signal: np.ndarray) -> float:
        """Calculate predictability index using local prediction error."""
        if len(signal) < 20:
            return np.nan
            
        errors = []
        for i in range(10, len(signal) - 1):
            # Simple local linear prediction
            local_data = signal[i-10:i]
            if len(local_data) >= 2:
                slope = (local_data[-1] - local_data[-2])
                prediction = local_data[-1] + slope
                error = abs(signal[i] - prediction)
                errors.append(error)
        
        return np.mean(errors) / np.std(signal) if errors else np.nan

    def _estimate_delay(self, signal: np.ndarray) -> int:
        """Estimate optimal delay using first minimum of autocorrelation."""
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find first zero crossing or minimum
        for i in range(1, min(len(autocorr), 100)):
            if autocorr[i] < 0:
                return i
            if i > 1 and autocorr[i] > autocorr[i-1]:
                return i-1
        
        return 1

    def _create_validation_heatmap(
        self, 
        results: Dict[str, Dict[str, Dict[str, float]]], 
        save_path: str
    ):
        """Create comprehensive validation heatmap."""
        surrogate_methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
        metrics = [
            "lyapunov_exponent", "time_irreversibility", "sample_entropy",
            "correlation_dimension", "nonlinearity_index", "predictability"
        ]
        
        n_systems = len(results)
        fig, axes = plt.subplots(1, n_systems, figsize=(5*n_systems, 6))
        if n_systems == 1:
            axes = [axes]
        
        for idx, (system_name, system_results) in enumerate(results.items()):
            # Create p-value matrix
            p_matrix = np.full((len(surrogate_methods), len(metrics)), np.nan)
            
            for i, method in enumerate(surrogate_methods):
                if method in system_results:
                    for j, metric in enumerate(metrics):
                        if metric in system_results[method]:
                            p_matrix[i, j] = system_results[method][metric]
            
            # Create heatmap
            im = axes[idx].imshow(
                p_matrix, 
                cmap='RdYlBu_r', 
                aspect='auto',
                vmin=0, vmax=1,
                interpolation='nearest'
            )
            
            # Add text annotations
            for i in range(len(surrogate_methods)):
                for j in range(len(metrics)):
                    if not np.isnan(p_matrix[i, j]):
                        text = axes[idx].text(
                            j, i, f'{p_matrix[i, j]:.3f}',
                            ha="center", va="center",
                            color="black" if p_matrix[i, j] > 0.5 else "white",
                            fontsize=8
                        )
            
            axes[idx].set_xticks(range(len(metrics)))
            axes[idx].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            axes[idx].set_yticks(range(len(surrogate_methods)))
            axes[idx].set_yticklabels(surrogate_methods)
            axes[idx].set_title(f'{system_name.replace("_", " ").title()}\nP-values (Observed ≠ Surrogate)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('P-value', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nValidation results saved to {save_path}")

    # ------------------------------------------------------------------
    # Individual surrogate algorithms (same as before but optimized)
    # ------------------------------------------------------------------
    def _ft(self) -> np.ndarray:
        """Phase-randomised Fourier surrogate."""
        x = self._detrend(self.signal)
        fftx = _fft.rfft(x)
        phases = self._rng.uniform(0, 2 * np.pi, len(fftx))
        fftx_randomised = np.abs(fftx) * np.exp(1j * phases)
        surrogate = _fft.irfft(fftx_randomised, n=len(x))
        return surrogate.astype(np.float64)

    def _aaft(self) -> np.ndarray:
        """Amplitude adjusted Fourier surrogate."""
        x = self._detrend(self.signal)
        ranks = x.argsort().argsort()
        gaussian = self._rng.normal(size=len(x))
        gaussian_sorted = np.sort(gaussian)
        x_gauss = gaussian_sorted[ranks]
        fftx = _fft.rfft(x_gauss)
        phases = self._rng.uniform(0, 2 * np.pi, len(fftx))
        x_phase = _fft.irfft(np.abs(fftx) * np.exp(1j * phases), n=len(x))
        surrogate = np.sort(x)[x_phase.argsort().argsort()]
        return surrogate.astype(np.float64)

    def _iaaft(self, *, n_iter: int = 100) -> np.ndarray:
        """Iterative amplitude adjusted Fourier surrogate."""
        x = self._detrend(self.signal)
        amplitude_target = np.sort(x)
        fft_target = np.abs(_fft.rfft(x))

        y = self._rng.permutation(x)
        for _ in range(n_iter):
            yf = _fft.rfft(y)
            y = _fft.irfft(fft_target * np.exp(1j * np.angle(yf)), n=len(x))
            y = amplitude_target[y.argsort().argsort()]
        return y.astype(np.float64)

    def _idfs(self, *, n_iter: int = 50) -> np.ndarray:
        """Iterative digitally-filtered shuffled surrogate."""
        x = self._detrend(self.signal)
        xp = self._rng.permutation(x)
        target_fft = np.abs(_fft.rfft(xp))

        y = amplitude_target = np.sort(x)
        for _ in range(n_iter):
            yf = _fft.rfft(y)
            y = _fft.irfft(target_fft * np.exp(1j * np.angle(yf)), n=len(x))
            y = amplitude_target[y.argsort().argsort()]
        return y.astype(np.float64)

    def _wiaaft(
        self,
        *,
        wavelet: str = "db4",
        level: int | None = None,
        n_iter: int = 20,
    ) -> np.ndarray:
        """Wavelet iterative amplitude adjusted Fourier surrogate."""
        x = self._detrend(self.signal)
        coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
        coeff_sur = []
        for c in coeffs[:-1]:
            csurr = self._iaaft_array(c, n_iter=n_iter)
            csurp = csurr[::-1]
            if np.linalg.norm(c - csurr) > np.linalg.norm(c - csurp):
                csurr = csurp
            coeff_sur.append(csurr)
        coeff_sur.append(coeffs[-1])
        y = pywt.waverec(coeff_sur, wavelet, mode="periodization")
        return y.astype(np.float64)[: len(x)]

    def _pps(
        self,
        *,
        tau: int | None = None,
        dim: int = 10,
        noise_factor: float | None = None,
    ) -> np.ndarray:
        """Pseudo-periodic surrogate."""
        x = self._detrend(self.signal)
        if tau is None:
            tau = self._first_zero_crossing_acf(x)
        embed = self._embed(x, dim, tau)
        dists = np.linalg.norm(embed - embed[-1], axis=1)
        q = np.argmin(dists[: len(embed) // 2])
        if noise_factor is None:
            noise_factor = 0.7 * np.mean(np.min(
                np.linalg.norm(embed[:-1, None] - embed[None, :-1], axis=2),
                axis=1,
            ))
        sur = [embed[self._rng.integers(len(embed))]]
        while len(sur) < len(embed):
            candidate = sur[-1] + self._rng.normal(scale=noise_factor, size=dim)
            j = np.argmin(np.linalg.norm(embed - candidate, axis=1))
            if j == len(embed) - 1:
                j = q
            sur.append(embed[(j + 1) % len(embed)])
        return np.asarray(sur)[:, 0].astype(np.float64)[: len(x)]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _detrend(self, x: np.ndarray) -> np.ndarray:
        return x - x.mean()

    def _iaaft_array(self, arr: np.ndarray, n_iter: int) -> np.ndarray:
        ranks = np.sort(arr)
        fft_target = np.abs(_fft.rfft(arr))
        y = self._rng.permutation(arr)
        for _ in range(n_iter):
            yf = _fft.rfft(y)
            y = _fft.irfft(fft_target * np.exp(1j * np.angle(yf)), n=len(arr))
            y = ranks[y.argsort().argsort()]
        return y

    def _embed(self, x: np.ndarray, dim: int, tau: int) -> np.ndarray:
        N = len(x) - (dim - 1) * tau
        return np.column_stack([x[i : i + N] for i in range(0, dim * tau, tau)])

    def _first_zero_crossing_acf(self, x: np.ndarray) -> int:
        acf = np.correlate(x, x, mode="full")[len(x) - 1 :]
        acf /= acf[0]
        zero_crossings = np.where(acf < 0)[0]
        return int(zero_crossings[0]) if len(zero_crossings) > 0 else 1

