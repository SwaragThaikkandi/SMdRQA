=================================================================
Chapter 4- Recurrence Plot and Recurrence Quantification Analysis
=================================================================

Eckmann et al. (1987) introduced a fascinating tool that helps us visualize the recurrence of states :math:`\vec{x_i}` in a phase space. Normally, a phase space lacks a visual dimension (such as two or three dimensions) that would allow us to picture it directly. To visualize higher-dimensional phase spaces, we often resort to projecting them onto two or three-dimensional sub-spaces.

However, Eckmann's tool offers us a unique perspective. It enables us to explore the trajectory of the :math:`m`-dimensional phase space through a two-dimensional representation of its recurrences. These recurrences capture instances where a state at time :math:`i` reappears at a different time :math:`j`. We depict these recurrences using a two-dimensional squared matrix filled with black and white dots (ones and zeros), with both axes representing time. This graphical representation is famously known as a recurrence plot (RP).

Mathematically, we can express an RP as follows:

.. math::

    R_{i,j} = \Theta(\varepsilon_i - \| \vec{x}_i - \vec{x}_j \|), \quad \vec{x}_i \in \mathbb{R}^m, \quad i,j = 1, \ldots, N

where :math:`N` represents the number of states :math:`x_i` under consideration, :math:`\varepsilon_i` is the threshold distance,:math:`\| \cdot \|` denotes a norm, and :math:`\Theta(\cdot)` represents the Heaviside function.


For RP, we can get measures that would quantify recurrence:

.. _table-recurrence-measures:

=============================================
Measure         | Formula                        | Description
=============================================
Recurrence Rate (RR) | \(RR = \frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N} R(i,j)\) | Measures the density of recurrence points in the recurrence plot. It indicates how likely it is for states to recur in the system, providing insights into system behavior and stability.
---------------------------------------------
Determinism (DET) | \(DET = \frac{\sum_{l=l_{\text{min}}}^{N}l\cdot p(l)}{\sum_{l=1}^{N}l\cdot p(l)}\) | Measures the fraction of diagonal line lengths that exceed a specified minimum line length. It assesses the predictability of the system, with higher values indicating more predictable behavior.
---------------------------------------------
Laminarity (LAM) | \(LAM = \frac{\sum_{v=v_{\text{min}}}^{N}v\cdot p(v)}{\sum_{v=1}^{N}v\cdot p(v)}\) | Similar to determinism but focuses on vertical (or horizontal) line lengths. It quantifies the extent to which the system remains trapped in a given state for an extended period, reflecting stability or periodic behavior.
---------------------------------------------
Average diagonal line length (L) | \(L = \frac{\sum_{l=l_{\text{min}}}^{N}l\cdot p(l)}{\sum_{l_{\text{min}}}^{N}p(l)}\) | Represents the average duration of diagonal lines in the recurrence plot. It provides a measure of how far into the future the system's behavior can be predicted based on its past states.
---------------------------------------------
Average vertical line length (Trapping time) (TT) | \(TT = \frac{\sum_{v=v_{\text{min}}}^{N}v\cdot p(v)}{\sum_{v_{\text{min}}}^{N}p(v)}\) | Indicates the average duration of vertical (or horizontal) lines in the recurrence plot. It reflects how long the system remains in a specific state before transitioning to a different state.
---------------------------------------------
Maximum diagonal line length | \(max(D_{l})\) where, \(D_{l} = \{l_{1}, l_{2}, l_{3}, ..., l_{Nd} \}\). \(Nd\) is the cardinality of the set \(D_{l}\) | Represents the maximum duration among all diagonal lines in the recurrence plot. It highlights the longest recurring patterns or periods in the system's behavior.
---------------------------------------------
Maximum vertical line length | \(max(D_{v})\) where, \(D_{v} = \{v_{1}, v_{2}, v_{3}, ..., v_{Nv} \}\). \(Nv\) is the cardinality of the set \(D_{l}\) | Represents the maximum duration among all vertical (or horizontal) lines in the recurrence plot. It signifies the longest periods of unchanged or slowly changing states in the system.
---------------------------------------------
Diagonal and Vertical Entropy (ENTR) | \(ENTR = -\sum_{l=l_{\text{min}}}^{N} p(l) \ln p(l)\) | Quantifies the degree of uncertainty in the system's states based on the distribution of diagonal and vertical line lengths in the recurrence plot. Higher entropy values indicate higher complexity and unpredictability in the system.
---------------------------------------------
