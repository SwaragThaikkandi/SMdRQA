"""
Rössler Attractor Tutorial
==========================

This example demonstrates two regimes of the Rössler system:

  1. Chaotic regime (a < 0.2)
  2. Periodic regime (a > 0.2)

We will:

  * Introduce the system equations.
  * Plot 3D trajectories for both a-values.
  * Generate FT, AAFT, IAAFT, IDFS, WIAAFT, and PPS surrogates.
  * Compute and compare the largest Lyapunov exponent and time-irreversibility.
  * Build and display recurrence plots for both signals.
  * Compute quantitative RQA measures and show how they differ.

Imports and setup:

"""

import numpy as np
import matplotlib.pyplot as plt

from SMdRQA.RQA2 import RQA2, RQA2_simulators, RQA2_tests

# Parameters (can be overridden via environment in CI)
N_SURR = int( os.getenv("N_SURR", 200) )
A_CHAOS = float(os.getenv("A_CHAOS", 0.1))
A_PER   = float(os.getenv("A_PER", 0.3))
B       = 0.2
C       = 5.7
TM      = 8000
N       = 2000

# ------------------------------------------------------------------------
# 1. Generate Rössler trajectories in two regimes
# ------------------------------------------------------------------------
sim = RQA2_simulators(seed=42)

x_chaos, y_chaos, z_chaos = sim.rossler(tmax=TM, n=N,
                                        a=A_CHAOS, b=B, c=C)
x_per,   y_per,   z_per   = sim.rossler(tmax=TM, n=N,
                                        a=A_PER,   b=B, c=C)

# ------------------------------------------------------------------------
# 2. Plot 3D attractors side by side
# ------------------------------------------------------------------------
fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(x_chaos, y_chaos, z_chaos, color="#d62728", lw=0.7)
ax1.set_title(f"Chaotic (a={A_CHAOS})")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(x_per, y_per, z_per, color="#1f77b4", lw=0.7)
ax2.set_title(f"Periodic (a={A_PER})")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

plt.tight_layout()

# ------------------------------------------------------------------------
# 3. Surrogate tests on x-component
# ------------------------------------------------------------------------
methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
metrics = ["lyapunov_exponent", "time_irreversibility"]

def compute_metrics(signal):
    tester = RQA2_tests(signal, seed=123, max_workers=4)
    # Generate surrogates
    S = tester.generate(kind=method, n_surrogates=N_SURR)
    # Compute metrics on original
    orig = tester._calculate_all_metrics(signal)
    # Compute metrics on surrogates
    distr = {m: [] for m in metrics}
    for s in S:
        vals = tester._calculate_all_metrics(s)
        for m in metrics:
            distr[m].append(vals[m])
    return orig, distr

results = {}
for regime, x in [("Chaotic", x_chaos), ("Periodic", x_per)]:
    results[regime] = {}
    for method in methods:
        o, d = compute_metrics(x)
        results[regime][method] = (o, d)

# ------------------------------------------------------------------------
# 4. Plot surrogate metric comparisons
# ------------------------------------------------------------------------
for regime in results:
    fig, axes = plt.subplots(1, len(metrics), figsize=(8,4))
    fig.suptitle(f"{regime} regime surrogate tests")
    for i, m in enumerate(metrics):
        ax = axes[i]
        for method in methods:
            _, dist = results[regime][method]
            ax.hist(dist[m], bins=30, alpha=0.3, label=method)
            ax.axvline(results[regime][method][0][m], color='k', lw=1)
        ax.set_title(m)
    axes[-1].legend(loc="upper right")
    plt.tight_layout(rect=[0,0,1,0.9])

# ------------------------------------------------------------------------
# 5. Recurrence plots and RQA metrics
# ------------------------------------------------------------------------
for regime, x in [("Chaotic", x_chaos), ("Periodic", x_per)]:
    rq = RQA2(x, dim=None, tau=None, eps=None, theiler=1)
    stats = rq.compute()
    # Build full distance matrix RP
    rp = rq._distance_matrix() <= rq._choose_threshold(rq._distance_matrix())
    plt.figure(figsize=(4,4))
    plt.imshow(rp, cmap="binary", origin="lower")
    plt.title(f"{regime} RP\nRR={stats['RR']:.3f}, DET={stats['DET']:.3f}")
    plt.tight_layout()
