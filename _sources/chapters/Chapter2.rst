====================================================
Chapter 2- Taken's Theorm and Time Delayed Embedding
====================================================

Since we have discussed phase space and its role in analyzing nonlinear systems, let's discuss about Taken's theorm. 

In 1981, Floris Takens published the paper “Detecting Strange Attractors in Turbulence”, which introduced this concept. Takens demonstrated that when a system involves multiple interconnected variables driving its dynamics (i.e., multidimensional dynamics), and if we can only observe a single variable :math:`x` from the system (i.e., measuring one dimension), then we can reconstruct the multidimensional dynamics of the system. This is achieved by plotting the observable :math:`x` against itself at specific time intervals and delays (refer to Figure 1). The process begins with the measured values of the variable :math:`x`:

.. math::

    \mathbf{x} = (x_1, x_2, x_3, \ldots, x_n)

Let :math:`\mathbf{x}` be a vector containing values :math:`x_1` to :math:`x_n`, representing the time-series of the variable :math:`x` sampled at regular intervals :math:`t_1, t_1 + \delta t, t_1 + 2\delta t, \ldots, t_1 + (n - 1)\delta t`. If we are aware of (or can estimate) the true dimension :math:`D` of the dynamical system from which we have sampled :math:`x`, then we can form :math:`D`-dimensional vectors in the following manner:

.. math::

    \boldsymbol{V_{1}} = (x_{1}, x_{1+\tau}, x_{1+2\tau}, \ldots, x_{1+(D-1)\tau})

The elements of :math:`\mathbf{V_{1}}` are derived from the vector :math:`\mathbf{x}`, starting with :math:`x_{1}` sampled at time :math:`t_{1}`, and then including values at later times, such as :math:`x_{1 + \tau}` sampled at :math:`t_{1} + \tau\delta t`. Here, :math:`\tau` represents the time-lag as the later times are delayed relative to :math:`t_{1}` by an integer multiple of :math:`\tau\delta t`.

A similar vector :math:`\mathbf{V_{2}}` can be constructed by starting with :math:`x_{2}` sampled at :math:`t_{2} = t_{1} + \delta t`. In fact, we can construct :math:`n - (D - 1)\tau` such vectors, which can then be arranged in a matrix.


.. math::

    \boldsymbol{V} = \begin{bmatrix}
                        \boldsymbol{V_{1}} \\
                        \boldsymbol{V_{2}} \\
                        \vdots \\
                        \boldsymbol{V_{n-(D-1)\tau}}
                      \end{bmatrix} = 
                      \begin{bmatrix}
                          x_{1}, x_{1+\tau}, x_{1+2\tau}, \ldots, x_{1+(D-1)\tau} \\
                          x_{2}, x_{2+\tau}, x_{2+2\tau}, \ldots, x_{2+(D-1)\tau} \\
                          x_{3}, x_{3+\tau}, x_{3+2\tau}, \ldots, x_{3+(D-1)\tau} \\
                          \vdots \\
                          x_{n-(D-1)\tau}, x_{n-(D-2)\tau}, x_{n-(D-3)\tau}, \ldots, x_{n}
                      \end{bmatrix}
    

FInding :math:`\tau` value
--------------------------

For practical purpose it is important to compute the appropriate value of the the delay(:math:`\tau`) in the first place. For this we had a multidimensional time series in which we com- puted a multidimensional mutual information and used it’s first minima(and global minima,in case the first minima doesn’t exist) in a plot between time delay and mutual information.

Let the time series be :math:`x_{n}` having length N The time delayed versions are given by:

.. math::

    x_{n}^{(0)} = x_{n}[1 : N-\tau]

.. math::

    x_{n}^{(\tau)} = x_{n}[\tau : N]

And the mutual information is given by:

.. math::

    I(x_{n}^{(0)}, x_{n}^{(\tau)}) = H(x_{n}^{(\tau)}) - [H(x_{n}^{(0)}, x_{n}^{(\tau)}) - H(x_{n}^{(0)})] = H(x_{n}^{(0)}) + H(x_{n}^{(\tau)}) - H(x_{n}^{(0)}, x_{n}^{(\tau)})

Where :math:`H` denoted entropy. 

We Implemented the Mutual information calculation as follows:

*mutualinfo()*
""""""""""""
.. autofunction:: SMdRQA.RQA_functions.mutualinfo

Note that, here in the source code, np.histogramdd is a function to compute d-dimensional histogram in numpy module. This function might lead to numpy core memory error if the input is of higher dimension

And then we compute the mutual information for a specific time delay by using the following function:

*timedelayMI()*
"""""""""""""""""""
.. autofunction:: SMdRQA.RQA_functions.timedelayMI

For understanding this function, let's consider the case of a simpe sine wave, as depicted below:

.. code-block:: python

   from SMdRQA.RQA_functions import timedelayMI
   import numpy as np
   import matplotlib.pyplot as plt

   angles = np.linspace(0,10*np.pi, 1000)

   angles = np.reshape(angles,(len(angles),1))

   sign = np.sin(angles)

   plt.figure(figsize = (12,9))
   plt.plot(range(len(sign)),sign,'r')
   plt.xlabel('time')
   plt.ylabel('signal')
   plt.title('signal')
   plt.show()



.. image:: https://raw.githubusercontent.com/SwaragThaikkandi/SMdRQA/main/docs/chapters/figs/simple-signal.svg
    :width: 800px
    :align: center


Now, let's compute the mutual inormation for different value of :math:`\tau` from 1 to n-1

.. code-block:: python 

   MI = []
   TAU = []
   for tau in range(len(angles)-1):
     TAU.append(tau)
     MI.append(timedelayMI(sign,len(sign),1, tau))

   plt.figure(figsize = (12,9))
   plt.plot(TAU, MI)
   plt.xlabel('$\\tau$')
   plt.ylabel('mutual information')
   plt.show()

.. image:: https://raw.githubusercontent.com/SwaragThaikkandi/SMdRQA/main/docs/chapters/figs/simple-signal-tau-mi.svg
    :width: 800px
    :align: center

We can see 10 peaks in this plot, but, it is not easy to understand what that codes for. For that we need to get the angular values corresponding to each of these :math:`\tau` values. 

.. code-block:: python 

   angles_from_tau = np.array(TAU)*((10*np.pi)/1000)
   angles_2pi = angles_from_tau % (2*np.pi)
   plt.figure(figsize = (12,9))
   plt.figure()
   plt.plot(angles_2pi, MI,'b.')
   plt.axvline(x = np.pi)
   plt.xlabel('angle')
   plt.ylabel('mutual information')
   plt.show

.. image:: https://raw.githubusercontent.com/SwaragThaikkandi/SMdRQA/main/docs/chapters/figs/simple-signal-tau-mi-angle.svg
    :width: 800px
    :align: center

We can see that the odd multiples of :math:`\frac{\pi}{2}` gives the minumum value of mutual information, and this is easy to understand in the case of sine wave as a sin wave shufted by :math:`\frac{\pi}{2}` or its multiples results in a cos wave. **findtau** function, in this context, finds the first minima of the mutual information vs :math:`\tau` curve. 

*timedelayMI()*
"""""""""""""""""""
.. autofunction:: SMdRQA.RQA_functions.findtau
