```markdown
# RQA2 Module Example - Rössler Attractor Analysis Tutorial

## Overview
This documentation walks you through the script and its output figures. The notebook-style layout interleaves explanatory text, Python code blocks, console output, and embedded figures to enable end-to-end reproducibility and understanding of nonlinear dynamics metrics.

### Prerequisites
- Python ≥ 3.8 with `numpy`, `matplotlib`, `seaborn`, `tqdm`
- *SMdRQA* package providing `RQA2`, `RQA2_simulators`, and `RQA2_tests`

---

## The Rössler System in Brief
The Rössler system is a three-dimensional continuous-time flow defined by:
$$
\dot x = -y - z,\\
\dot y = x + a y,\\
\dot z = b + z(x - c).
$$
Here, $x$ and $y$ act as transverse coordinates that spiral in the $x$–$y$ plane, while $z$ feeds back nonlinearly, providing a “kick” whenever $x$ passes the threshold set by the parameter $c$.  The constants $a$ and $b$ control linear growth or decay in the $y$ and $z$ directions, respectively, and $c$ regulates the strength of the nonlinear stretching in the $z$‐equation.

For the canonical choice of parameters $a=b=0.2$ and $c=5.7$, the system exhibits a strange (chaotic) attractor.  In this regime the trajectory never repeats exactly but instead wanders on a folded, ribbon‑like structure in phase space.  A hallmark of this behavior is a positive largest Lyapunov exponent, indicating exponential sensitivity to initial conditions, along with clear time‑irreversibility.  By contrast, if $a$ is reduced below approximately 0.1 (keeping $b=0.2$ and $c=5.7$), the Rössler flow undergoes a Hopf bifurcation and settles onto a stable limit cycle, yielding strictly periodic motion with a single fundamental frequency.

Geometrically, one may understand the Rössler attractor as follows.  In the $x$–$y$ plane the combined action of $\dot x=-y-z$ and $\dot y=x+a\,y$ produces a weakly repelling spiral when $a>0$.  As the orbit spirals outward, the term $z\,(x-c)$ in $\dot z$ remains small until $x$ exceeds $c$.  At that point $z$ grows rapidly, and the term $-z$ in the $\dot x$‐equation “snaps” the trajectory back toward the origin.  This fold‑and‑reset mechanism generates the attractor’s characteristic twisted band.

As one increases $a$ from zero to about 0.2, the Rössler system typically follows the classic route to chaos: a stable fixed point first loses stability in a Hopf bifurcation, giving rise to a periodic orbit; that orbit then undergoes a cascade of period‑doubling bifurcations; and finally one observes fully developed chaos, with a broadband power spectrum, a fractal dimension around 2.0–2.1, and positive Lyapunov exponent.

Key diagnostics for studying the Rössler attractor include (1) the largest Lyapunov exponent $\lambda_1>0$, which measures the rate at which nearby trajectories diverge; (2) the fractal (correlation) dimension $D_2\approx2.05$, which quantifies the attractor’s “thickness”; and (3) Poincaré sections (for example, sampling the flow whenever $z=c$), which reveal a Cantor‑set–like 1D return map.  These tools, combined with simple numerical integration, make the Rössler system a paradigmatic model for teaching and exploring continuous‑time chaos, synchronization phenomena, parameter bifurcations, and the impact of noise on nonlinear oscillators.

| **a-value** | **Behavior**                 |
|-------------|------------------------------|
| 0.30        | Chaotic band-fold attractor  |
| 0.10        | Period-1 limit cycle         |

---

## Analysis Pipeline
Five-step workflow:

1. Simulate trajectories  
2. Visualize phase space  
3. Generate surrogate data  
4. Compute RQA measures  
5. Summarize key results  

---

### 1. Generating Trajectories
```python
N  = 2000     # Samples after subsampling
TM = 8000     # Integration steps
B, C = 0.2, 5.7
A_CHAOS, A_PER = 0.3, 0.1

sim = RQA2_simulators(seed=42)
x_chaos, y_chaos, z_chaos = sim.rossler(tmax=TM, n=N, a=A_CHAOS, b=B, c=C)
x_per, y_per, z_per = sim.rossler(tmax=TM, n=N, a=A_PER, b=B, c=C)
```

---

### 2. 3-D Phase Portraits
![Chaotic vs periodic Rössler attractors](rossler_3d_attractors.png)  
*Left*: Chaotic banded attractor. *Right*: Periodic closed orbit.

---

### 3. Surrogate Testing
Tests six null hypotheses: `FT`, `AAFT`, `IAAFT`, `IDFS`, `WIAAFT`, `PPS`.

```python
methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
metrics = ["lyapunov_exponent", "time_irreversibility"]
N_SURR = 200

def compute_surrogate_metrics(signal, method):
    tester = RQA2_tests(signal, seed=123, max_workers=2)
    surrogates = tester.generate(kind=method, n_surrogates=N_SURR)
    orig_metrics = tester._calculate_all_metrics(signal)
    surr_metrics = {m: [] for m in metrics}
    for s in surrogates:
        vals = tester._calculate_all_metrics(s)
        for m in metrics:
            surr_metrics[m].append(vals[m])
    return orig_metrics, surr_metrics
```

#### Results
![Surrogate test results](surrogate_results.png)  
**Top row**: Chaotic regime ($a=0.2$).  
**Bottom row**: Periodic regime ($a=0.05$).  

**Interpretation**:

Surrogate data testing is a nonparametric hypothesis‐testing method used to decide whether a measured time series exhibits genuine nonlinear or deterministic structure, as opposed to being generated by a linear stochastic process or simple periodic oscillation.  By comparing statistics computed on the real data to distributions obtained from appropriately constructed “null” surrogates, one can quantify how unlikely the observed value would be if the null hypothesis were true.

#### Null Hypotheses and Surrogate Types

1. **Fourier‐transform (FT) surrogates** preserve the power spectrum but randomize all phases, thus enforcing linear Gaussian structure.
2. **Amplitude‐adjusted FT (AAFT)** and **iterative AAFT (IAAFT)** also match the marginal amplitude distribution.
3. **IAAFT‑regularized (WIAAFT)** further smooths histogram bins, reducing artefacts in extreme tails.
4. **Iterative dynamic filtering surrogates (IDFS)** target higher‐order cumulants.
5. **Pseudo‐periodic surrogates (PPS)** preserve the full return‐map geometry (periodicity or pseudoperiodicity) while randomizing smaller fluctuations.

#### Metrics under Test

* **Largest Lyapunov Exponent** $\lambda_1$: measures average exponential divergence of nearby trajectories; a positive value indicates chaos.  We plot $\log_{10}(\lambda_1)$.
* **Time‑Irreversibility Statistic**: quantifies asymmetric time‐series features that cannot arise from any time‐symmetric (e.g. linear) process; also displayed on a log scale.

#### Chaotic Regime (top row)

* **Largest Lyapunov Exponent (top‐left):**  The true Rössler exponent (dashed line at $\log_{10}\lambda_1\approx -2.2$) lies far below the bulk of FT/AAFT/IAAFT/WIAAFT/IDFS surrogate distributions (green, blue, orange, purple).  These null models destroy the low‐dimensional flow and create effectively high‐dimensional noise, inflating divergence rates (surrogate $\log_{10}\lambda_1\sim -1.3$ to $-1.0$).  Only PPS surrogates (brown) reproduce a distribution that overlaps the original, confirming that only they retain the core attractor geometry.
* **Time‑Irreversibility (top‐right):**  The real data’s irreversibility statistic (dashed line at $\approx2.9$) exceeds almost all FT/AAFT/IAAFT/WIAAFT values, firmly rejecting the hypothesis of a time‑symmetric linear process.  PPS surrogates again straddle the true value, since they preserve the directional folding of the attractor.

#### Periodic Regime (bottom row)

* **Largest Lyapunov Exponent (bottom‐left):**  For periodic Rössler (a=0.05), the true exponent is near zero (dashed at $\log_{10}\lambda_1\approx -1.75$).  FT/AAFT/IAAFT/WIAAFT/IDFS surrogates generate spurious positive estimates (clusters around $-1.2$ to $-0.9$), reflecting destroyed periodicity.  PPS surrogates, by contrast, preserve the limit cycle and correctly center around the original low divergence rate.
* **Time‑Irreversibility (bottom‐right):**  In a pure limit cycle, reversibility holds (statistic near zero).  Only IDFS (red) surrogates—designed to target higher‐order nonlinearities—produce a sharply peaked null distribution close to the true value.  Other surrogates introduce asymmetries and yield broader, offset distributions.

### Conclusion

1. **Rejection of Linear Nulls in Chaos:**  In the chaotic Rössler regime, all linear surrogates (FT family and IDFS) fail to match the observed low Lyapunov exponent and high time‐irreversibility, providing clear evidence of low‐dimensional deterministic chaos.

2. **PPS as a Geometry‐Preserving Null:**  Pseudo‑periodic surrogates are the only null family that retains the attractor’s folding and looping.  Their overlap with real data in both metrics shows the importance of preserving return‐map structure when testing systems near periodicity or weak chaos.

3. **Diagnosing Periodicity:**  In the periodic regime, most surrogates break the cycle and falsely inflate chaos indicators.  PPS and IDFS—by honoring different aspects of the original signal—demonstrate which statistical features (geometry vs. higher‑order moments) are critical.

Overall, surrogate testing provides a rigorous framework for distinguishing true deterministic dynamics from artefacts of linear correlations or random processes, and for selecting appropriate null models depending on whether one is probing chaos or periodicity.

---

### 4. Recurrence Quantification Analysis (RQA)
Embedding parameters selected automatically:
- **Delay (τ)**: First minimum of Mutual Information.
- **Dimension (m)**: False Nearest Neighbors (FNN) drops to 0%.

```python
rq = RQA2(data=signal, normalize=True, reqrr=0.05, lmin=2)
measures = rq.compute_rqa_measures()
```

#### Embedding Parameter Selection
| **Figure** | **Description**                                  |
|------------|--------------------------------------------------|
| ![τ for chaotic](Chaotic_tau_mi_plot.png) | Mutual Information vs. τ (chaotic) |
| ![m for chaotic](Chaotic_fnn_curve_plot.png) | FNN vs. m (chaotic)            |
| ![τ for periodic](Periodic_tau_mi_plot.png) | Mutual Information vs. τ (periodic) |
| ![m for periodic](Periodic_fnn_curve_plot.png) | FNN vs. m (periodic)          |

Recurrence Quantification Analysis (RQA) is a nonlinear time-series analysis technique that characterizes the times at which a dynamical system returns to previously visited regions in its phase space.  Since real-world measurements often provide only a single scalar time series $x(t)$, reconstructing an equivalent representation of the system’s full state space is a critical preliminary step.  Takens’ embedding theorem guarantees that, under mild conditions, a time-delay embedding of the form

$$
\mathbf{X}(t) = \bigl\[,x(t),,x(t+\tau),,x(t+2\tau),,\dots,,x(t+(m-1)\tau)\bigr]
$$

in an $m$-dimensional space is diffeomorphic (one-to-one and smooth) to the original attractor, provided that the embedding dimension $m$ is sufficiently large ($m>2d_f$, where $d_f$ is the fractal dimension) and the delay $\tau$ avoids redundancy.

##### Choosing the Time Delay $\tau$

The time delay $\tau$ determines the spacing between successive coordinates in the delay vector.  Two competing requirements must be balanced:

1. **Statistical independence:**  If $\tau$ is too small, successive coordinates $x(t)$ and $x(t+\tau)$ are highly correlated, causing the reconstructed manifold to lie near the diagonal hyperplane, wasting dimensions.

2. **Dynamic relevance:**  If $\tau$ is too large, the coordinates become effectively independent and the reconstruction may sample points from unrelated regions of the attractor, destroying the local geometry.

A standard method for selecting $\tau$ is to compute the average mutual information

$$
I\[\tau] = \sum\_{i,j} p\_{ij}(\tau) \log \frac{p\_{ij}(\tau)}{p\_i p\_j}
$$

where $p_i=\Pr(x(t)\in\text{bin }i)$ and $p_{ij}(\tau)=\Pr(x(t)\in i,\,x(t+\tau)\in j)$.  The first local minimum of $I[\tau]$ (Figure \:ref:`fig:tau-mi`) identifies the delay $\tau^*\approx27$ at which coordinates share minimal redundant information yet remain causally linked by the system’s evolution.

##### Selecting the Embedding Dimension $m$

The embedding dimension $m$ must be large enough to unfold the attractor so that distinct trajectories in the original phase space do not project onto the same point in the reconstructed space.  The False Nearest Neighbors (FNN) algorithm quantitatively assesses this requirement:

1. For each candidate $m$, compute the nearest neighbor distance $R_m(i)$ between points $\mathbf{X}_m(t_i)$ and its nearest neighbor in $\mathbb{R}^m$.
2. Calculate the distance in $m+1$ dimensions by appending the next delayed coordinate $x(t+(m)\tau)$.  If the increase in distance exceeds a threshold (relative to $R_m(i)$), the neighbor is classified as false.
3. The ratio of false neighbors over all points yields the FNN fraction at dimension $m$.

In Figure \:ref:`fig:fnn`, the FNN fraction decreases sharply from nearly 1.0 at $m=1$ to essentially zero at $m=5$.  The first $m$ at which the FNN ratio falls below a small tolerance (e.g., 1–2%) is chosen as the optimal embedding dimension; here, $m^*=5$ ensures a one-to-one unfolding of the Rössler attractor.

##### Implications for RQA

With $\tau^*=27$ and $m^*=5$, the delay-coordinate vectors

$$
\mathbf{X}(t) = \bigl\[x(t),,x(t+27),,x(t+2\cdot27),,x(t+3\cdot27),,x(t+4\cdot27)\bigr]
$$

span a reconstructed phase space that accurately preserves the geometry and topology of the original Rössler attractor.  Consequently:

* **Recurrence plots** constructed by thresholding $\|\mathbf{X}(t_i)-\mathbf{X}(t_j)\|$ reveal true return times and recurrence structures.
* **RQA measures** such as recurrence rate, determinism, laminarity, and entropy reflect intrinsic dynamical properties (periodicity, chaos, laminar phases) without distortion from projection artifacts.

Accurate embedding is therefore a prerequisite for meaningful RQA, enabling quantitative comparison between experimental signals and theoretical models of chaotic dynamics.



#### Key RQA Metrics
| Metric | Meaning                                                                 |
|--------|-------------------------------------------------------------------------|
| RR     | Recurrence Rate (density of recurrence points)                          |
| DET    | Determinism (share of points forming diagonal lines > *l_min*)         |
| L      | Mean diagonal line length (predictability horizon)                      |
| Lmax   | Longest diagonal line (inverse sensitivity ↔ $1/\lambda_{\max}$)       |
| DIV    | Divergence = $1/\text{Lmax}$                                           |
| LAM    | Laminarity (proportion of points on vertical lines)                     |

**Console Output**:
```
Chaotic RQA measures:
  recurrence_rate          : 0.0500
  determinism              : 0.8735
  average_diagonal_length  : 20.37
  max_diagonal_length      : 122

Periodic RQA measures:
  recurrence_rate          : 0.0500
  determinism              : 0.9967
  average_diagonal_length  : 1985.00
  max_diagonal_length      : 1998
```

---

### 5. Recurrence Plots
![Recurrence plots](recurrence_plots.png)  
- **Chaotic**: Broken diagonals → high unpredictability.
- **Periodic**: Uninterrupted diagonals → near-perfect determinism.

---

## Final Analysis Summary
```
============================================================
ANALYSIS SUMMARY
============================================================

System Parameters:
  Chaotic regime: a=0.3, b=0.2, c=5.7
  Periodic regime: a=0.1, b=0.2, c=5.7
  Time series length: 2000 points
  Surrogates per method: 200

Surrogate Methods Tested: FT, AAFT, IAAFT, IDFS, WIAAFT, PPS
Nonlinear Metrics: lyapunov_exponent, time_irreversibility

RQA Results Comparison:
Measure                   Chaotic      Periodic    
--------------------------------------------------
recurrence_rate           0.0454       0.0459      
determinism               0.9996       1.0000      
average_diagonal_length   48.2231      168.3731    
max_diagonal_length       1892.0000    1898.0000   
```

### Interpretation
- **RR** fixed at ~5% by adaptive thresholding.
- **DET** ≈1.0 for periodic orbit → strict periodicity.
- **L/Lmax** explode in periodic regime → precise state revisits.

---

## Metric Reference Sheet
### Surrogate Diagnostics
- `lyapunov_exponent`: Largest Lyapunov exponent (Wolf–Sano algorithm).
- `time_irreversibility`: Ramsey & Rothman bicovariance statistic.

### RQA Core Measures
| Metric | Formula/Description                                  |
|--------|------------------------------------------------------|
| RR     | $RR = \frac{1}{N^2}\sum_{i,j} R_{ij}$               |
| DET    | % recurrence points in diagonals ≥ *l_min*           |
| L/Lmax | Mean/max diagonal line length                        |
| DIV    | $1 / \text{Lmax}$                                    |
| LAM    | % recurrence points in vertical lines                |
| TT     | Mean vertical line length (trapping time)            |

---

## Output Files
- `rossler_3d_attractors.png`
- `surrogate_results.png`
- `recurrence_plots.png`

---

## Further Reading
- Strogatz, *Nonlinear Dynamics and Chaos*
- Marwan et al., *Physics Reports* 438 (2007)
- Marwan, *Int. J. Bifurcation & Chaos* 21 (2011)

---

## Appendix A – Full Script Listing
```python:linenos
# [Full script content from rossler_attractor_analysis.py]
```

---

## References
1. Wolf et al., *Physica D* (1985)  
2. Ramsey & Rothman, *Econometrics* (1996)  
3. Marwan et al., *Physics Reports* 438 (2007)  
4. Kennel et al., *Phys. Rev. A* (1992)  
5. Marwan, *Int. J. Bifurcation & Chaos* 21 (2011)  
```
