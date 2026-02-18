#!/usr/bin/env python3

# rossler_attractor_analysis.py - Corrected version based on actual RQA2.py implementation
# Set up matplotlib for non-interactive use

from tqdm import tqdm
from SMdRQA.RQA2 import RQA2, RQA2_simulators, RQA2_tests
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# Import from the actual RQA2 module structure

print("Starting Rössler Attractor Analysis...")

# Section 1: System Setup & Trajectory Generation
N = 2000
TM = 8000
B = 0.2
C = 5.7
A_CHAOS = 0.3  # Chaotic regime (a > 0.2)
A_PER = 0.1    # Periodic/synchronous regime (a < 0.2)

print("Generating Rössler trajectories...")
sim = RQA2_simulators(seed=42)

# Generate chaotic and periodic trajectories
x_chaos, y_chaos, z_chaos = sim.rossler(tmax=TM, n=N, a=A_CHAOS, b=B, c=C)
x_per, y_per, z_per = sim.rossler(tmax=TM, n=N, a=A_PER, b=B, c=C)

print(f"Generated trajectories: {len(x_chaos)} points each")

# Section 2: 3D Attractor Visualization
print("Creating 3D visualizations...")
fig = plt.figure(figsize=(15, 6))

# Chaotic attractor
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_chaos, y_chaos, z_chaos, c='maroon', lw=0.5, alpha=0.7)
ax1.set_title(
    f"Chaotic Rössler Attractor (a={A_CHAOS})",
    fontweight='bold',
    fontsize=14)
ax1.set_xlabel("X", fontsize=12)
ax1.set_ylabel("Y", fontsize=12)
ax1.set_zlabel("Z", fontsize=12)
ax1.view_init(30, -60)

# Periodic attractor
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_per, y_per, z_per, c='teal', lw=0.5, alpha=0.7)
ax2.set_title(
    f"Periodic Rössler Attractor (a={A_PER})",
    fontweight='bold',
    fontsize=14)
ax2.set_xlabel("X", fontsize=12)
ax2.set_ylabel("Y", fontsize=12)
ax2.set_zlabel("Z", fontsize=12)
ax2.view_init(30, -60)

plt.tight_layout()
plt.savefig('rossler_3d_attractors.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: rossler_3d_attractors.png")

# Section 3: Surrogate Analysis
N_SURR = 200  # Reduced for faster computation in demo
methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
metrics = ["lyapunov_exponent", "time_irreversibility"]


def compute_surrogate_metrics(signal, method):
    """Compute metrics for original signal and its surrogates."""
    print(f"    Processing {method} surrogates...")
    tester = RQA2_tests(signal, seed=123, max_workers=2)

    # Generate surrogates
    surrogates = tester.generate(kind=method, n_surrogates=N_SURR)

    # Compute metrics for original signal
    orig_metrics = tester._calculate_all_metrics(signal)

    # Compute metrics for all surrogates
    surr_metrics = {m: [] for m in metrics}
    for i, s in tqdm(enumerate(surrogates)):
        if i % 10 == 0:
            print(f"      Surrogate {i+1}/{N_SURR}")
        s_vals = tester._calculate_all_metrics(s)
        for m in metrics:
            surr_metrics[m].append(s_vals[m])
            print('Metric value:', s_vals[m])

    return orig_metrics, surr_metrics


# Compute surrogate metrics for both regimes
results = {}
print("\nStarting surrogate analysis...")

for regime, signal in [("Chaotic", x_chaos), ("Periodic", x_per)]:
    results[regime] = {}
    print(
        f"\nProcessing {regime} regime (a={'0.1' if regime=='Chaotic' else '0.3'})...")

    for method in methods:
        try:
            orig, surr = compute_surrogate_metrics(signal, method)
            results[regime][method] = (orig, surr)
        except Exception as e:
            print(f"      Error with {method}: {e}")
            # Create dummy data to avoid plotting errors
            orig = {m: np.nan for m in metrics}
            surr = {m: [np.nan] * N_SURR for m in metrics}
            results[regime][method] = (orig, surr)

# Section 4: Surrogate Results Visualization
print("\nCreating surrogate analysis plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = plt.cm.tab10.colors

for r, regime in enumerate(["Chaotic", "Periodic"]):
    for m, metric in enumerate(metrics):
        ax = axes[r, m]

        # Plot histograms for each surrogate method
        for i, method in enumerate(methods):
            if method in results[regime]:
                orig_val, surr_data = results[regime][method]

                # Filter out NaN values
                valid_surr = [x for x in surr_data[metric] if not np.isnan(x)]

                if valid_surr:
                    sns.kdeplot(data=np.log(valid_surr),
                                ax=ax,
                                label=method,
                                fill=True,            # fill under the curve
                                alpha=0.6,            # transparency
                                color=colors[i % len(colors)],
                                linewidth=2)

                    # Plot original value as vertical line
                    if not np.isnan(orig_val[metric]):
                        ax.axvline(np.log(orig_val[metric]), color=colors[i % len(
                            colors)], linestyle='--', linewidth=2, alpha=0.8)

        ax.set_title(f"{regime}: {metric.replace('_', ' ').title()}",
                     fontweight='bold', fontsize=12)
        ax.set_xlabel("Log Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)

# Add legend to the last subplot
axes[1, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)

plt.suptitle(
    "Surrogate Test Results: Original (dashed lines) vs Surrogate Distributions",
    fontsize=14,
    fontweight='bold')
plt.tight_layout()
plt.savefig('surrogate_results.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: surrogate_results.png")

# Section 5: Recurrence Quantification Analysis
print("\nComputing RQA measures...")


def compute_rqa_analysis(signal, regime_name):
    """Compute RQA measures for a given signal."""
    print(f"  Processing {regime_name} regime...")

    # Create RQA object with appropriate parameters
    # Use the actual constructor parameters from RQA2.py
    rq = RQA2(data=signal, normalize=True, reqrr=0.05, lmin=2)

    # Compute RQA measures
    measures = rq.compute_rqa_measures()

    rq.plot_tau_mi_curve(save_path=regime_name + '_tau_mi_plot.png')

    rq.plot_fnn_curve(save_path=regime_name + '_fnn_curve_plot.png')

    # Get the recurrence plot
    rp = rq.recurrence_plot

    return rq, measures, rp


# Analyze both regimes
rqa_results = {}
for regime, signal in [("Chaotic", x_chaos), ("Periodic", x_per)]:
    try:
        rq, measures, rp = compute_rqa_analysis(signal, regime)
        rqa_results[regime] = {
            'rq_object': rq,
            'measures': measures,
            'recurrence_plot': rp
        }

        # Print key measures
        print(f"    {regime} RQA measures:")
        for key, value in measures.items():
            if key in [
                'recurrence_rate',
                'determinism',
                'average_diagonal_length',
                    'max_diagonal_length']:
                print(f"      {key}: {value:.4f}")

    except Exception as e:
        print(f"    Error computing RQA for {regime}: {e}")
        rqa_results[regime] = None

# Section 6: Recurrence Plot Visualization
print("\nCreating recurrence plots...")
fig = plt.figure(figsize=(14, 6))

for i, (regime, signal) in enumerate(
        [("Chaotic", x_chaos), ("Periodic", x_per)]):
    ax = fig.add_subplot(1, 2, i + 1)

    if regime in rqa_results and rqa_results[regime] is not None:
        rp = rqa_results[regime]['recurrence_plot']
        measures = rqa_results[regime]['measures']

        # Display recurrence plot
        ax.imshow(rp, cmap='binary', origin='lower', aspect='equal')

        # Create title with key measures
        title = (f"{regime} Recurrence Plot\n"
                 f"RR={measures['recurrence_rate']:.3f}, "
                 f"DET={measures['determinism']:.3f}\n"
                 f"L={measures['average_diagonal_length']:.2f}, "
                 f"Lmax={measures['max_diagonal_length']:.0f}")

        ax.set_title(title, fontweight='bold', fontsize=11)
    else:
        ax.set_title(
            f"{regime} - Analysis Failed",
            fontweight='bold',
            fontsize=11)
        ax.text(0.5, 0.5, "RQA computation failed", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Time Index", fontsize=10)
    ax.set_ylabel("Time Index", fontsize=10)

plt.tight_layout()
plt.savefig('recurrence_plots.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: recurrence_plots.png")

# Section 7: Summary Report
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(f"\nSystem Parameters:")
print(f"  Chaotic regime: a={A_CHAOS}, b={B}, c={C}")
print(f"  Periodic regime: a={A_PER}, b={B}, c={C}")
print(f"  Time series length: {N} points")
print(f"  Surrogates per method: {N_SURR}")

print(f"\nSurrogate Methods Tested: {', '.join(methods)}")
print(f"Nonlinear Metrics: {', '.join(metrics)}")

if rqa_results:
    print(f"\nRQA Results Comparison:")
    print(f"{'Measure':<25} {'Chaotic':<12} {'Periodic':<12}")
    print("-" * 50)

    key_measures = [
        'recurrence_rate',
        'determinism',
        'average_diagonal_length',
        'max_diagonal_length']

    for measure in key_measures:
        chaos_val = rqa_results.get(
            'Chaotic',
            {}).get(
            'measures',
            {}).get(
            measure,
            np.nan)
        period_val = rqa_results.get(
            'Periodic',
            {}).get(
            'measures',
            {}).get(
            measure,
            np.nan)
        print(f"{measure:<25} {chaos_val:<12.4f} {period_val:<12.4f}")

print(f"\nFiles generated:")
print("  - rossler_3d_attractors.png")
print("  - surrogate_results.png")
print("  - recurrence_plots.png")

print(f"\nAnalysis completed successfully!")
print("=" * 60)
