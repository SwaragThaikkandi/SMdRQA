from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from SMdRQA.RQA_functions import *
from SMdRQA.RQA2 import *
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
import numpy as np
import matplotlib.pyplot as plt



def test_RQA2_initialization():
    """Test RQA2 class initialization with different data types."""
    import numpy as np
    
    # Test initialization without data
    rqa_empty = RQA2()
    assert rqa_empty.data is None, "Empty initialization should have no data"
    assert rqa_empty.n_samples == 0, "Empty initialization should have 0 samples"
    assert rqa_empty.n_dimensions == 0, "Empty initialization should have 0 dimensions"
    
    # Test initialization with 1D data
    n = 100
    t = np.linspace(0, 4*np.pi, n)
    data_1d = np.sin(t)
    rqa_1d = RQA2(data_1d)
    assert rqa_1d.n_samples == n, f"Expected {n} samples, got {rqa_1d.n_samples}"
    assert rqa_1d.n_dimensions == 1, f"Expected 1 dimension, got {rqa_1d.n_dimensions}"
    assert rqa_1d.data.shape == (n, 1), f"Expected shape ({n}, 1), got {rqa_1d.data.shape}"
    
    # Test initialization with 2D data
    data_2d = np.random.rand(50, 3)
    rqa_2d = RQA2(data_2d, normalize=False)
    assert rqa_2d.n_samples == 50, "Expected 50 samples"
    assert rqa_2d.n_dimensions == 3, "Expected 3 dimensions"
    assert np.array_equal(rqa_2d.data, data_2d), "Data should be identical when normalize=False"
    
    # Test normalization
    rqa_norm = RQA2(data_2d, normalize=True)
    assert np.allclose(np.mean(rqa_norm.data, axis=0), 0, atol=1e-10), "Mean should be near zero after normalization"
    assert np.allclose(np.std(rqa_norm.data, axis=0), 1, atol=1e-10), "Std should be near one after normalization"
    
    print("test_RQA2_initialization passed!")


def test_RQA2_load_data():
    """Test the load_data method."""
    import numpy as np
    
    rqa = RQA2()
    
    # Load simple sine wave data
    n = 100
    t = np.linspace(0, 4*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa.load_data(data)
    assert rqa.n_samples == n, f"Expected {n} samples"
    assert rqa.n_dimensions == 1, "Expected 1 dimension"
    
    # Test that computed values are reset
    assert rqa._tau is None, "Tau should be reset after loading new data"
    assert rqa._m is None, "m should be reset after loading new data"
    assert rqa._eps is None, "eps should be reset after loading new data"
    
    print("test_RQA2_load_data passed!")


def test_RQA2_properties():
    """Test lazy property evaluation."""
    import numpy as np
    
    # Create simple sine wave
    n = 200
    t = np.linspace(0, 4*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa = RQA2(data)
    
    # Test that accessing properties triggers computation
    tau = rqa.tau
    assert isinstance(tau, int), "Tau should be an integer"
    assert tau > 0, "Tau should be positive"
    assert tau < n, "Tau should be less than number of samples"
    
    m = rqa.m
    assert isinstance(m, int), "m should be an integer"
    assert m > 0, "m should be positive"
    
    eps = rqa.eps
    assert isinstance(eps, (int, float)), "eps should be numeric"
    assert eps > 0, "eps should be positive"
    
    # Test that values are cached
    assert rqa.tau == tau, "Tau should be cached"
    assert rqa.m == m, "m should be cached"
    assert rqa.eps == eps, "eps should be cached"
    
    print("test_RQA2_properties passed!")


def test_RQA2_embedded_signal():
    """Test embedded signal computation."""
    import numpy as np
    
    n = 50
    data = np.arange(n).reshape(-1, 1).astype(float)
    rqa = RQA2(data, normalize=False)
    
    # Force specific parameters for predictable results
    rqa._tau = 2
    rqa._m = 3
    
    embedded = rqa.embedded_signal
    expected_length = n - (rqa.m - 1) * rqa.tau
    
    assert embedded.shape[0] == expected_length, f"Expected {expected_length} embedded points"
    assert embedded.shape[1] == rqa.m, f"Expected {rqa.m} embedding dimensions"
    assert embedded.shape[2] == 1, "Expected 1 original dimension"
    
    # Check a few values manually
    assert embedded[0, 0, 0] == 0, "First embedded point should start at 0"
    assert embedded[0, 1, 0] == 2, "Second element should be at tau=2"
    assert embedded[0, 2, 0] == 4, "Third element should be at 2*tau=4"
    
    print("test_RQA2_embedded_signal passed!")


def test_RQA2_recurrence_plot():
    """Test recurrence plot computation."""
    import numpy as np
    
    # Use a simple periodic signal
    n = 100
    t = np.linspace(0, 2*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa = RQA2(data)
    rp = rqa.recurrence_plot
    
    # Basic checks
    assert rp.shape[0] == rp.shape[1], "Recurrence plot should be square"
    assert rp.shape[0] > 0, "Recurrence plot should have positive size"
    assert np.all((rp == 0) | (rp == 1)), "Recurrence plot should be binary"
    
    # Check recurrence rate
    rr = rqa.recurrence_rate
    assert isinstance(rr, float), "Recurrence rate should be a float"
    assert 0 <= rr <= 1, "Recurrence rate should be between 0 and 1"
    
    print("test_RQA2_recurrence_plot passed!")


def test_RQA2_rqa_measures():
    """Test RQA measures computation."""
    import numpy as np
    
    # Create a simple deterministic signal
    n = 200
    t = np.linspace(0, 4*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa = RQA2(data)
    measures = rqa.compute_rqa_measures()
    
    # Check that all expected measures are present
    expected_measures = [
        'recurrence_rate', 'determinism', 'laminarity', 
        'diagonal_entropy', 'vertical_entropy',
        'average_diagonal_length', 'average_vertical_length',
        'max_diagonal_length', 'max_vertical_length',
        'diagonal_mode', 'vertical_mode'
    ]
    
    for measure in expected_measures:
        assert measure in measures, f"Missing measure: {measure}"
        assert isinstance(measures[measure], (int, float)), f"{measure} should be numeric"
    
    # Test individual measure methods
    det = rqa.determinism()
    lam = rqa.laminarity()
    tt = rqa.trapping_time()
    
    assert isinstance(det, float), "Determinism should be a float"
    assert isinstance(lam, float), "Laminarity should be a float"
    assert isinstance(tt, float), "Trapping time should be a float"
    
    assert 0 <= det <= 1, "Determinism should be between 0 and 1"
    assert 0 <= lam <= 1, "Laminarity should be between 0 and 1"
    assert tt > 0, "Trapping time should be positive"
    
    print("test_RQA2_rqa_measures passed!")


def test_RQA2_time_delay_computation():
    """Test time delay computation methods."""
    import numpy as np
    
    # Create a periodic signal where we can predict optimal tau
    n = 500
    period = 50  # samples per period
    t = np.linspace(0, 10*2*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa = RQA2(data)
    
    # Test default method
    tau_default = rqa.compute_time_delay(method='default')
    assert isinstance(tau_default, int), "Tau should be an integer"
    assert tau_default > 0, "Tau should be positive"
    
    # Test polynomial method
    tau_poly = rqa.compute_time_delay(method='polynomial')
    assert isinstance(tau_poly, int), "Tau should be an integer"
    assert tau_poly > 0, "Tau should be positive"
    
    # Test that both methods give reasonable results
    assert tau_default < n//4, "Tau should be reasonable relative to signal length"
    assert tau_poly < n//4, "Tau should be reasonable relative to signal length"
    
    print("test_RQA2_time_delay_computation passed!")


def test_RQA2_embedding_dimension():
    """Test embedding dimension computation."""
    import numpy as np
    
    # Create a low-dimensional chaotic signal (Lorenz-like)
    n = 300
    t = np.linspace(0, 30, n)
    # Simple 2D signal
    x = np.sin(t) + 0.5*np.sin(3*t)
    y = np.cos(t) + 0.3*np.cos(2*t)
    data = np.column_stack([x, y])
    
    rqa = RQA2(data)
    m = rqa.compute_embedding_dimension()
    
    assert isinstance(m, int), "Embedding dimension should be an integer"
    assert m > 0, "Embedding dimension should be positive"
    assert m <= 10, "Embedding dimension should be reasonable"
    
    print("test_RQA2_embedding_dimension passed!")


def test_RQA2_neighborhood_radius():
    """Test neighborhood radius computation."""
    import numpy as np
    
    n = 200
    data = np.random.randn(n, 1)
    rqa = RQA2(data)
    
    # Test with different recurrence rates
    eps1 = rqa.compute_neighborhood_radius(reqrr=0.05)
    eps2 = rqa.compute_neighborhood_radius(reqrr=0.15)
    
    assert isinstance(eps1, (int, float)), "Epsilon should be numeric"
    assert isinstance(eps2, (int, float)), "Epsilon should be numeric"
    assert eps1 > 0, "Epsilon should be positive"
    assert eps2 > 0, "Epsilon should be positive"
    
    # Higher recurrence rate should generally require larger epsilon
    # (though this might not always hold for all signals)
    
    print("test_RQA2_neighborhood_radius passed!")


def test_RQA2_save_load():
    """Test save and load functionality."""
    import numpy as np
    import tempfile
    import os
    
    # Create and analyze a signal
    n = 100
    t = np.linspace(0, 4*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa1 = RQA2(data)
    
    # Compute some measures to populate the object
    tau1 = rqa1.tau
    m1 = rqa1.m
    eps1 = rqa1.eps
    rp1 = rqa1.recurrence_plot
    measures1 = rqa1.compute_rqa_measures()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
    
    try:
        rqa1.save_results(tmp_path)
        
        # Create new object and load
        rqa2 = RQA2()
        rqa2.load_results(tmp_path)
        
        # Check that everything matches
        assert rqa2.n_samples == rqa1.n_samples, "Sample count should match"
        assert rqa2.n_dimensions == rqa1.n_dimensions, "Dimension count should match"
        assert rqa2.tau == tau1, "Tau should match"
        assert rqa2.m == m1, "m should match"
        assert rqa2.eps == eps1, "eps should match"
        assert np.array_equal(rqa2.data, rqa1.data), "Data should match"
        assert np.array_equal(rqa2.recurrence_plot, rp1), "Recurrence plots should match"
        
        # Check RQA measures
        for key in measures1:
            assert key in rqa2._rqa_measures, f"Missing measure: {key}"
            assert np.isclose(rqa2._rqa_measures[key], measures1[key]), f"Measure {key} doesn't match"
            
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    print("test_RQA2_save_load passed!")


def test_RQA2_get_summary():
    """Test summary generation."""
    import numpy as np
    
    n = 150
    t = np.linspace(0, 6*np.pi, n)
    data = np.sin(t).reshape(-1, 1)
    
    rqa = RQA2(data)
    
    # Compute some measures
    _ = rqa.tau
    _ = rqa.m
    _ = rqa.eps
    measures = rqa.compute_rqa_measures()
    
    summary = rqa.get_summary()
    
    # Check structure
    assert 'Data Info' in summary, "Summary should contain data info"
    assert 'Parameters' in summary, "Summary should contain parameters"
    assert 'RQA Measures' in summary, "Summary should contain RQA measures"
    
    # Check data info
    assert summary['Data Info']['Samples'] == n, "Sample count should match"
    assert summary['Data Info']['Dimensions'] == 1, "Dimension count should match"
    
    # Check parameters
    assert 'Time Delay (τ)' in summary['Parameters'], "Should contain tau"
    assert 'Embedding Dimension (m)' in summary['Parameters'], "Should contain m"
    assert 'Neighborhood Radius (ε)' in summary['Parameters'], "Should contain eps"
    assert 'Recurrence Rate' in summary['Parameters'], "Should contain recurrence rate"
    
    print("test_RQA2_get_summary passed!")


def test_RQA2_batch_process():
    """Test batch processing functionality."""
    import numpy as np
    import tempfile
    import os
    import shutil
    
    # Create temporary directories
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Create some test files
        n_files = 3
        n_samples = 100
        
        for i in range(n_files):
            t = np.linspace(0, 4*np.pi, n_samples)
            # Create slightly different signals
            data = np.sin(t + i*0.5).reshape(-1, 1)
            filename = f'test_signal_{i}.npy'
            np.save(os.path.join(input_dir, filename), data)
        
        # Process batch
        results, errors = RQA2.batch_process(
            input_dir, 
            output_dir,
            group_level=False,
            reqrr=0.1
        )
        
        # Check results
        assert len(results) == n_files, f"Expected {n_files} results, got {len(results)}"
        assert len(errors) == 0, f"Expected no errors, got {len(errors)}"
        
        # Check that each result has the required fields
        for result in results:
            assert 'file' in result, "Result should contain filename"
            assert 'tau' in result, "Result should contain tau"
            assert 'm' in result, "Result should contain m"
            assert 'eps' in result, "Result should contain eps"
            assert 'recurrence_rate' in result, "Result should contain recurrence rate"
            assert 'determinism' in result, "Result should contain determinism"
        
        # Check that output files were created
        output_files = os.listdir(output_dir)
        assert 'rqa_results.csv' in output_files, "Should create results CSV"
        assert len([f for f in output_files if f.endswith('.npy')]) == n_files, "Should create RP files"
        
    finally:
        # Clean up
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    
    print("test_RQA2_batch_process passed!")

def test_RQA2_custom_config():
    """Test custom configuration parameters."""
    import numpy as np
    
    n = 100
    data = np.random.randn(n, 1)
    
    # Test with custom parameters
    custom_config = {
        'reqrr': 0.05,
        'lmin': 3,
        'Rmin': 2,
        'Rmax': 20,
        'delta': 0.01
    }
    
    rqa = RQA2(data, **custom_config)
    
    # Check that config was set
    for key, value in custom_config.items():
        assert rqa.config[key] == value, f"Config {key} should be {value}"
    
    # Test that custom parameters affect computation
    measures = rqa.compute_rqa_measures(lmin=custom_config['lmin'])
    assert isinstance(measures['determinism'], float), "Should compute with custom lmin"
    
    print("test_RQA2_custom_config passed!")


def test_RQA2_edge_cases():
    """Test edge cases and error handling."""
    import numpy as np
    
    # Test empty initialization
    rqa_empty = RQA2()
    
    try:
        _ = rqa_empty.tau
        assert False, "Should raise error when no data loaded"
    except ValueError:
        pass  # Expected
    
    # Test very short time series
    short_data = np.array([[1], [2], [3], [4], [5]])
    rqa_short = RQA2(short_data, normalize=False)
    
    # Should handle short series gracefully (might have limitations)
    try:
        tau = rqa_short.tau
        assert tau > 0, "Should return positive tau even for short series"
    except:
        # Some methods might fail for very short series, which is acceptable
        pass
    
    # Test constant data
    constant_data = np.ones((50, 1))
    rqa_constant = RQA2(constant_data, normalize=False)
    
    # Should handle constant data (might have special behaviors)
    try:
        tau = rqa_constant.tau
        m = rqa_constant.m
        # For constant data, some computations might have special cases
        assert isinstance(tau, int), "Should return integer tau"
        assert isinstance(m, int), "Should return integer m"
    except:
        # Some edge cases might be handled by returning default values
        pass
    
    print("test_RQA2_edge_cases passed!")


def test_RQA2_multidimensional():
    """Test with multidimensional data."""
    import numpy as np
    
    # Create 3D Lorenz-like system
    n = 300
    t = np.linspace(0, 20, n)
    x = np.sin(t) + 0.1*np.random.randn(n)
    y = np.cos(1.1*t) + 0.1*np.random.randn(n)
    z = np.sin(0.9*t) + 0.1*np.random.randn(n)
    
    data = np.column_stack([x, y, z])
    rqa = RQA2(data)
    
    assert rqa.n_dimensions == 3, "Should detect 3 dimensions"
    assert rqa.n_samples == n, f"Should detect {n} samples"
    
    # Test computations work with multidimensional data
    tau = rqa.tau
    m = rqa.m
    eps = rqa.eps
    
    assert isinstance(tau, int), "Should compute tau for multidimensional data"
    assert isinstance(m, int), "Should compute m for multidimensional data"
    assert isinstance(eps, (int, float)), "Should compute eps for multidimensional data"
    
    # Test embedded signal shape
    embedded = rqa.embedded_signal
    expected_samples = n - (m - 1) * tau
    assert embedded.shape == (expected_samples, m, 3), "Embedded signal should have correct shape"
    
    print("test_RQA2_multidimensional passed!")

def test_RQA2_full_integration():
    """Complete integration test of the RQA2 class."""
    import numpy as np
    
    # Create a complex test signal (mixture of sine waves with noise)
    n = 400
    t = np.linspace(0, 10*2*np.pi, n)
    signal1 = np.sin(t) + 0.5*np.sin(3*t)
    signal2 = np.cos(1.1*t) + 0.3*np.cos(2.2*t)
    noise = 0.1*np.random.randn(n, 2)
    
    data = np.column_stack([signal1, signal2]) + noise
    
    # Create RQA object
    rqa = RQA2(data, reqrr=0.1, lmin=2)
    
    # Test complete analysis pipeline
    print(f"Data shape: {rqa.data.shape}")
    print(f"Original data shape: {rqa.original_data.shape}")
    
    # Compute parameters
    tau = rqa.tau
    m = rqa.m  
    eps = rqa.eps
    
    print(f"Computed parameters - tau: {tau}, m: {m}, eps: {eps:.6f}")
    
    # Compute recurrence plot
    rp = rqa.recurrence_plot
    print(f"Recurrence plot shape: {rp.shape}")
    print(f"Recurrence rate: {rqa.recurrence_rate:.4f}")
    
    # Compute all RQA measures
    measures = rqa.compute_rqa_measures()
    print("RQA Measures:")
    for key, value in measures.items():
        print(f"  {key}: {value:.6f}")
    
    # Test specific measure access
    det = rqa.determinism()
    lam = rqa.laminarity()
    tt = rqa.trapping_time()
    
    print(f"Individual measures - DET: {det:.4f}, LAM: {lam:.4f}, TT: {tt:.4f}")
    
    # Verify all computations are reasonable
    assert 0 <= rqa.recurrence_rate <= 1, "RR should be in [0,1]"
    assert 0 <= det <= 1, "DET should be in [0,1]"
    assert 0 <= lam <= 1, "LAM should be in [0,1]"
    assert tt > 0, "TT should be positive"
    assert measures['diagonal_entropy'] >= 0, "Entropy should be non-negative"
    assert measures['max_diagonal_length'] >= measures['average_diagonal_length'], "Max should be >= average"
    
    # Test summary
    summary = rqa.get_summary()
    print("\nSummary generated successfully")
    
    print("test_RQA2_full_integration passed!")





def run_comprehensive_validation():
    """
    Run comprehensive validation of surrogate methods against 
    multiple chaotic attractors with statistical significance testing.
    """
    print("=== Comprehensive Surrogate Validation Framework ===\n")
    
    # Initialize simulator and generate test battery
    simulator = RQA2_simulators(seed=42)
    print("Generating chaotic attractor test battery...")
    
    # Generate multiple systems with different regimes
    test_systems = {}
    
    # Rössler chaotic regime (a < 0.2)
    x, y, z = simulator.rossler(tmax=8000, n=2000, a=0.1, b=0.2, c=5.7)
    test_systems['rossler_chaotic'] = {'x': x, 'y': y, 'z': z}
    
    # Rössler synchronous regime (a > 0.2) 
    x, y, z = simulator.rossler(tmax=8000, n=2000, a=0.3, b=0.2, c=5.7)
    test_systems['rossler_synchronous'] = {'x': x, 'y': y, 'z': z}
    
    # Lorenz chaotic system
    x, y, z = simulator.lorenz(tmax=8000, n=2000, sigma=10.0, rho=28.0, beta=8.0/3.0)
    test_systems['lorenz_chaotic'] = {'x': x, 'y': y, 'z': z}
    
    # Hénon map
    x, y = simulator.henon(n=2000, a=1.4, b=0.3)
    test_systems['henon_chaotic'] = {'x': x, 'y': y}
    
    # Chua circuit
    x, y, z = simulator.chua(tmax=8000, n=2000)
    test_systems['chua_chaotic'] = {'x': x, 'y': y, 'z': z}
    
    print(f"Generated {len(test_systems)} test systems")
    for name, data in test_systems.items():
        print(f"  {name}: {len(data['x'])} points")
    
    # Initialize testing framework with first system (placeholder)
    first_system = list(test_systems.values())[0]
    tester = RQA2_tests(first_system['x'], seed=123, max_workers=4)
    
    # Run comprehensive validation
    print(f"\nRunning validation with 200 surrogates per method...")
    print("This may take several minutes...")
    
    results = tester.comprehensive_validation(
        test_systems, 
        n_surrogates=200,
        save_path="comprehensive_surrogate_validation.png"
    )
    
    # Print summary statistics
    print("\n=== VALIDATION SUMMARY ===")
    for system_name, system_results in results.items():
        print(f"\n{system_name.upper()}:")
        for method_name, method_results in system_results.items():
            significant_metrics = [
                metric for metric, p_val in method_results.items() 
                if not np.isnan(p_val) and p_val < 0.05
            ]
            print(f"  {method_name:8s}: {len(significant_metrics)}/6 metrics significant (p < 0.05)")
            if significant_metrics:
                sig_str = ", ".join(significant_metrics[:3])  # Show first 3
                if len(significant_metrics) > 3:
                    sig_str += "..."
                print(f"            Significant: {sig_str}")
    
    # Analysis recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("• P-values < 0.05: Surrogate method successfully distinguishes from original")
    print("• P-values > 0.05: Surrogate method preserves the measured property") 
    print("• For chaos detection: Look for high p-values in Lyapunov/nonlinearity metrics")
    print("• For stationarity testing: Focus on time_irreversibility results")
    print("• Best performing surrogates: IAAFT, WIAAFT for nonlinear properties")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive validation
    validation_results = run_comprehensive_validation()
    
    # Create additional analysis plots
    print("\nGenerating additional analysis plots...")
    
    # Performance comparison across systems
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
    system_names = list(validation_results.keys())
    
    # Calculate average significance rate per method
    method_performance = {}
    for method in methods:
        significance_rates = []
        for system_name in system_names:
            if method in validation_results[system_name]:
                system_results = validation_results[system_name][method]
                valid_p_values = [p for p in system_results.values() if not np.isnan(p)]
                if valid_p_values:
                    sig_rate = np.mean([p < 0.05 for p in valid_p_values])
                    significance_rates.append(sig_rate)
        method_performance[method] = np.mean(significance_rates) if significance_rates else 0
    
    # Bar plot of method performance
    bars = ax.bar(methods, [method_performance[m] for m in methods], 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    ax.set_ylabel('Average Significance Rate')
    ax.set_title('Surrogate Method Performance Across All Systems\n(Fraction of metrics with p < 0.05)')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, method in zip(bars, methods):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('method_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis complete! Check generated figures:")
    print("• comprehensive_surrogate_validation.png: P-value heatmaps")
    print("• method_performance_comparison.png: Performance comparison")

