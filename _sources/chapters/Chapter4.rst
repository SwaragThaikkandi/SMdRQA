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

.. container:: bullets

   **Recurrence Rate (RR)**
      - **Formula:** :math:`RR = \frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N} R(i,j)`
      - **Description:** RR measures the density of recurrence points in the recurrence plot (RP). It indicates how probable it is for states to recur in the system. RR is calculated as the average of :math:`R(i,j)` across the RP.

   **Determinism (DET)**
      - **Formula:** :math:`DET = \frac{\sum_{l=l_{\text{min}}}^{N}l\cdot p(l)}{\sum_{l=1}^{N}l\cdot p(l)}`
      - **Description:** DET quantifies what fraction of diagonal line lengths are above a minimum line length. Diagonal lines in the RP represent consecutive periods of recurrence, and DET reflects the predictability of the dynamical system based on these lines.

   **Laminarity (LAM)**
      - **Formula:** :math:`LAM = \frac{\sum_{v=v_{\text{min}}}^{N}v\cdot p(v)}{\sum_{v=1}^{N}v\cdot p(v)}`
      - **Description:** LAM is similar to DET but considers vertical (or horizontal) line lengths in the RP. It indicates the extent to which the system is trapped in a state for some time, reflecting its persistence.

   **Average Diagonal Line Length (L)**
      - **Formula:** :math:`L = \frac{\sum_{l=l_{\text{min}}}^{N}l\cdot p(l)}{\sum_{l_{\text{min}}}^{N}p(l)}`
      - **Description:** L represents the average value of the diagonal line length distribution in the RP. It quantifies how far in time the dynamical system is predictable based on the lengths of diagonal lines.

   **Average Vertical Line Length (Trapping Time) (TT)**
      - **Formula:** :math:`TT = \frac{\sum_{v=v_{\text{min}}}^{N}v\cdot p(v)}{\sum_{v_{\text{min}}}^{N}p(v)}`
      - **Description:** TT is the average value of the vertical line length distribution in the RP. It indicates the average time a state remains unchanged or changes very slowly in the system.

   **Maximum Diagonal Line Length**
      - **Formula:** :math:`\max(D_{l})`
      - **Description:** This represents the maximum value from the diagonal line length distribution in the RP, indicating the longest continuous period of recurrence in the system.

   **Maximum Vertical Line Length**
      - **Formula:** :math:`\max(D_{v})`
      - **Description:** This reflects the maximum value from the vertical line length distribution in the RP, indicating the longest time a state remains unchanged or changes very slowly.

   **Diagonal and Vertical Entropy (ENTR)**
      - **Formula:** :math:`ENTR = -\sum_{l=l_{\text{min}}}^{N} p(l) \ln p(l)`
      - **Description:** ENTR quantifies the degree of uncertainty in possible states and the complexity of the dynamical system. It uses the distributions of diagonal and vertical line lengths present in the RP to assess system complexity.
