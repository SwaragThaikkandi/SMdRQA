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

.. container::
   :name: table-1

   .. table:: Measures in Recurrence Analysis. For explanation about
   :math:`R(i,j)` refer equation: `[eq:4] <#eq:4>`__. Recurrence rate
   (RR) is the average of :math:`R(i,j)` across RP. :math:`l` stands for
   diagonal line length, which is the continuous section of RP, having
   value of :math:`R(i,j) = 1`.

      +----------------------+----------------------+----------------------+
      | **Measure**          | **Formula**          | **Description**      |
      +======================+======================+======================+
      | Recurrence Rate (RR) | :                    | Measures density of  |
      |                      | math:`RR = \frac{1}{ | recurrence points in |
      |                      | N^2}\sum_{i=1}^{N}\s | the recurrence plot, |
      |                      | um_{j=1}^{N} R(i,j)` | indicating how       |
      |                      |                      | probable recurrence  |
      |                      |                      | of states is in the  |
      |                      |                      | system.              |
      +----------------------+----------------------+----------------------+
      | Determinism (DET)    | :math:               | Measures what        |
      |                      | `DET = \frac{\sum_{l | fraction of the      |
      |                      | =l_{\text{min}}}^{N} | diagonal line        |
      |                      | l\cdot p(l)}{\sum_{l | lengths are above a  |
      |                      | =1}^{N}l\cdot p(l)}` | minimum, given, line |
      |                      |                      | length. :math:`p(l)` |
      |                      |                      | is the probability   |
      |                      |                      | of a line length     |
      |                      |                      | :math:`l`. Since     |
      |                      |                      | diagonal lines are   |
      |                      |                      | markers of           |
      |                      |                      | consecutive periods  |
      |                      |                      | of recurrence in the |
      |                      |                      | data, determinism    |
      |                      |                      | corresponds to the   |
      |                      |                      | predictability of    |
      |                      |                      | the dynamical        |
      |                      |                      | system.              |
      +----------------------+----------------------+----------------------+
      | Laminarity (LAM)     | :math:               | Mathematically       |
      |                      | `LAM = \frac{\sum_{v | equivalent to        |
      |                      | =v_{\text{min}}}^{N} | determinism but      |
      |                      | v\cdot p(v)}{\sum_{v | defined for vertical |
      |                      | =1}^{N}v\cdot p(v)}` | (or horizontal) line |
      |                      |                      | lengths. Since       |
      |                      |                      | vertical (or         |
      |                      |                      | horizontal) lines    |
      |                      |                      | are markers of       |
      |                      |                      | states that do not   |
      |                      |                      | change or change     |
      |                      |                      | very slowly,         |
      |                      |                      | laminarity           |
      |                      |                      | quantifies the       |
      |                      |                      | extent of the        |
      |                      |                      | dynamical system     |
      |                      |                      | being trapped in any |
      |                      |                      | given state for some |
      |                      |                      | time.                |
      +----------------------+----------------------+----------------------+
      | Average diagonal     | :math:`L             | Average value of     |
      | line length (L)      |  = \frac{\sum_{l=l_{ | diagonal line length |
      |                      | \text{min}}}^{N}l\cd | distribution,        |
      |                      | ot p(l)}{\sum_{l_{\t | quantifying how far  |
      |                      | ext{min}}}^{N}p(l)}` | in time the          |
      |                      |                      | dynamical system is  |
      |                      |                      | predictable.         |
      +----------------------+----------------------+----------------------+
      | Average vertical     | :math:`TT            | Average value of the |
      | line length (TT)     |  = \frac{\sum_{v=v_{ | vertical line length |
      |                      | \text{min}}}^{N}v\cd | distribution.        |
      |                      | ot p(v)}{\sum_{v_{\t |                      |
      |                      | ext{min}}}^{N}p(v)}` |                      |
      +----------------------+----------------------+----------------------+
      | Maximum diagonal     | :math:`\max(D_{l})`  | Maximum value from   |
      | line length          | where,               | the diagonal line    |
      |                      | :math:`D_{l}         | distribution         |
      |                      | = \{l_{1}, l_{2}, l_ |                      |
      |                      | {3}, ..., l_{Nd} \}` |                      |
      +----------------------+----------------------+----------------------+
      | Maximum vertical     | :math:`\max(D_{v})`  | Maximum value from   |
      | line length          | where,               | the vertical line    |
      |                      | :math:`D_{v}         | distribution         |
      |                      | = \{v_{1}, v_{2}, v_ |                      |
      |                      | {3}, ..., v_{Nv} \}` |                      |
      +----------------------+----------------------+----------------------+
      | Diagonal and         | :math:`ENTR = -\s    | Quantifies the       |
      | Vertical Entropy     | um_{l=l_{\text{min}} | degree of            |
      | (ENTR)               | }^{N} p(l) \ln p(l)` | uncertainty in the   |
      |                      |                      | possible states and  |
      |                      |                      | hence, the           |
      |                      |                      | complexity of the    |
      |                      |                      | dynamical system,    |
      |                      |                      | using the            |
      |                      |                      | distribution of      |
      |                      |                      | diagonal and         |
      |                      |                      | vertical line        |
      |                      |                      | lengths present in   |
      |                      |                      | the plot,            |
      |                      |                      | respectively.        |
      +----------------------+----------------------+----------------------+
      |                      |                      |                      |
      +----------------------+----------------------+----------------------+
