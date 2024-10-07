============================================================
Chapter 3- False Nearest Neighbours and Embedding Dimensions
============================================================

We employed a technique known as false nearest neighbors (FNN), introduced by Kennel et al.(1992), to ascertain the minimum number of dimensions needed to faithfully reconstruct an attractor. An attractor serves as a representation of how a system evolves over time.

In our analysis, our goal was to ensure that when we represent this evolution in lower dimensions, we do not lose crucial details. False nearest neighbors play a crucial role in this determination. They are points that may seem close together in a lower-dimensional depiction but are actually distant in the original higher-dimensional space.

To illustrate, consider trying to draw a detailed picture of a tree on a small piece of paper. You might find it challenging to capture all the branches and leaves due to space constraints. Similarly, in our analysis, starting with a lower-dimensional representation of the system's behavior may obscure important details. This situation is akin to trying to fit a detailed tree drawing on a tiny piece of paper – some parts may appear connected, but they are distinct.

Identifying many false nearest neighbors suggests that our lower-dimensional representation lacks essential information. Increasing the dimensions allows us to "unfold" the attractor and separate these falsely identified neighbors. This process ensures that our representation accurately captures the system's dynamics. Hence, false nearest neighbors serve as a guide to determine if our chosen dimensions adequately represent the complete picture of the system's behavior over time.

In simpler terms, let's say we have a time series with an ideal embedding dimension of :math:`m_0`. When we reduce the dimension by one, some points are strongly affected and become false nearest neighbors (FNN). To identify these FNN points, we compared the distances between points in the :math:`m`-dimensional space with those in the :math:`m+1`-dimensional space and calculated their ratio.

.. math::

    X_{fnn}(r) = \frac{\sum_{n=1}^{N-m-1} \Theta\left(\frac{|x_n^{(m+1)}-x_{k(n)}^{(m+1)}|}{|x_n^{(m)}-x_{k(n)}^{(m)}|}-r\right) \Theta\left(\frac{\sigma}{r}-|x_n^{(m)}-x_{k(n)}^{(m)}|\right)}{\sum_{n=1}^{N-m-1}\Theta\left(\frac{\sigma}{r}-|x_n^{(m)}-x_{k(n)}^{(m)}|\right)}

Where :math:` x_{k(n)}^{(m)}` represents the nearest neighbor of :math:`x_n^{(m)}` in :math:`m` dimensions, and :math:`\Theta(x)` denotes the Heaviside step function. :math:`\sigma` signifies the standard deviation of the data. The function :math:`k(n)` is defined as follows:

.. math::

    k(n) = \{ n' \mid |x_n^{(m)} - x_{n'}^{(m)}| \leq |x_n^{(m)} - x_{n''}^{(m)}|, \forall n'' \in I(x) - n \}

where :math:`I(x)` is the set of all indices of x. k(n) is estimated using **nearest()** function in the package

*nearest()*
""""""""""""
.. autofunction:: SMdRQA.RQA_functions.nearest

A more appropriate method would be to see the FNN ratio as a function of r (see Kantz and Schreiber(2004), section 3.3.1, page 37, figure 3.3)) if we are interested in parameter exploration. 

For this we defined the radius at which FNN hist zero as: 

.. math::

    r_0(m) = \{ r \mid X_{fnn}(r) < \delta, \forall r \in [r_{\min}, r_{\max}] \}

Where :math:`[r_{min},r_{max}]` is the interval where we are searching, which is for m embedding dimensions, and :math:`\delta` is a small enough number. This was implemented using the function **fnnhitszero()**

*fnnhitszero()*
""""""""""""
.. autofunction:: SMdRQA.RQA_functions.fnnhitszero

When doing this for different values of embedding dimensions, we will get r values which when ploted against corresponding embedding dimension would look like the following figure: 

.. image:: https://raw.githubusercontent.com/SwaragThaikkandi/SMdRQA/main/docs/chapters/figs/23_Embedding_Dim_Hitts_Zero.png
    :width: 800px
    :align: center

Here, we can see that, after a point, the curve shallows down, that is we can assume that the attractor had fairly unfolded. For getting this value of :math:`m`, we need to find the knee point of this monotonously decreasing curve. This could be done by choosing a termination criteria like the following:

.. math::

    m = \{ \text{max}(m') \mid r_{0}(m'-1) - r_{0}(m') \geq \beta \}


Which was implemented using the function **findm()**. 

*findm()*
"""""""""""""""""""
.. autofunction:: SMdRQA.RQA_functions.findm


Now we have two important parameters for time delayed embedding, :math:`tau` and :math:`m`. 
