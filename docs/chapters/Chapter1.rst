==============================
Chapter 1- The Nonlinear World
==============================

Dynamical Systems and Terminologies
-----------------------------------

There are two primary kinds of dynamical systems: those described by **differential equations** and those governed by **iterated maps**, also known as difference equations. **Differential equations** capture the behavior of systems evolving continuously in time, while **iterated maps** are applicable to scenarios where time progresses discretely.

Let's consider the example of a damped harmonic oscillator: 

.. math::

    m \frac{d^2 x}{dt^2} + b \frac{dx}{dt} + kx = 0

- \( m \) : Mass of the oscillator.
- \( b \) : Damping coefficient.
- \( k \) : Spring constant.
- \( x \) : Displacement of the oscillator from its equilibrium position.
- \( t \) : Time.

We call this as an ordinary differential equation as the equation only involves derivatives with respect to time. 

A general framework used to represent system of ordinary differential equation is:

.. math::

    \dot{x}_1 = f_1(x_1, x_2, x_3, \ldots, x_n)

    \dot{x}_2 = f_2(x_1, x_2, x_3, \ldots, x_n)

    \vdots

    \dot{x}_n = f_n(x_1, x_2, x_3, \ldots, x_n)

Let's consider an example to understand what this system of equations can mean, to simplify it and to reduce jargons. suppose given system of differential equations represents a set of equations describing the rates of change of three populations in an ecosystem: rabbits (x₁), foxes (x₂), and wolves (x₃).

- The first equation describes the rate of change of the rabbit population (dx₁/dt) as:

  .. math::

      \frac{dx_1}{dt} = \dot{x}_1 = f_1(x_1, x_2, x_3) = 0.1x_1 - 0.01x_2

  This equation indicates that the rabbit population increases by 0.1 rabbits per unit of time but decreases by 0.01 rabbits per unit of time for each fox present.

- The second equation represents the rate of change of the fox population (dx₂/dt) as:

  .. math::

      \frac{dx_2}{dt} = \dot{x}_2 = f_2(x_1, x_2, x_3) = 0.005x_1x_2 - 0.02x_2 - 0.01x_3

  This equation shows that the fox population increases by 0.005 times the product of rabbits and foxes but decreases by 0.02 foxes per unit of time and also by 0.01 foxes per unit of time for each wolf present.

- The third equation describes the rate of change of the wolf population (dx₃/dt) as:

  .. math::

      \frac{dx_3}{dt} = \dot{x}_3 = f_3(x_1, x_2, x_3) = 0.03x_2 - 0.02x_3

  This equation indicates that the wolf population increases by 0.03 wolves per unit of time but decreases by 0.02 wolves per unit of time for each wolf present.

In summary, these equations model how the populations of rabbits, foxes, and wolves change over time based on interactions within the ecosystem. Each equation captures the dynamics of one species and how it is influenced by the presence or absence of other species in the system.


Linear and Nonlinear Systems
----------------------------

Let's consider the case of damped oscillator

.. math::

    \ddot{x} = - \frac{b}{m} \dot{x} - \frac{k}{m} x 

Now, let's call (dx₁/dt) as x₂ and x as x₁

.. math::

    \dot{x_2} = \ddot{x_1} = - \frac{b}{m} x_2 - \frac{k}{m} x_1

Now, we have the equalent system:

.. math::

    \dot{x}_1 &= x_2 \\
    \dot{x}_2 &= -\frac{b}{m} x_2 - \frac{k}{m} x_1


The system is categorized as linear since all occurrences of \( x_i \) on the right-hand side are raised to the power of one only. Otherwise, the system would be considered nonlinear.

A good example of that would be that of a swinging pendulum:

.. math::

    \frac{d^2 \theta}{dt^2} = -\frac{g}{L} \sin(\theta)

- :math:`\frac{d^2 \theta}{dt^2}`: The second derivative of the angle :math:`\theta` with respect to time :math:`t`. This term represents the angular acceleration of the pendulum, indicating how quickly the angular velocity changes over time.

- :math:`g`: The acceleration due to gravity. It represents the force that pulls the pendulum downward towards the center of the Earth.

- :math:`L`: The length of the pendulum. It is the distance between the pivot point and the center of mass of the pendulum.

- :math:`\theta`: The angular displacement of the pendulum from the vertical position. It measures the angle between the pendulum's position and the vertical axis.

For this example, the equalent system is nonlinear. Let :math:`\theta_1` = :math:`\theta` and :math:`\theta_2` = :math:`\frac{d \theta}{dt}`

.. math::

    \dot{\theta_1} &= \theta_2 \\
    \dot{\theta_2} &= -\frac{g}{L} \sin(\theta_1)

Nonlinearity in the pendulum equation complicates analytical solutions. One common workaround is to use the small angle approximation, sin x = x for x << 1. This simplifies the problem to a linear one, making it easier to solve. However, this approximation excludes certain physical motions, like the pendulum swinging over the top, which are important.

Is it truly necessary to resort to such drastic approximations? Surprisingly, the pendulum equation can be solved analytically using elliptic functions. Nonetheless, we seek a simpler approach that captures the pendulum's behavior directly. At low energy, it swings back and forth, while at high energy, it whirls over the top. We aim to extract this information directly from the system using geometric methods. This is where we need the notion of a phase space!


Phase Space
-----------

The rough idea behind phase space is as follows:

- Suppose we know the solution for the pendulum system for a set of initial conditions: :math:`(\theta_{1}(0),\theta_{2}(0))` this represent the position and velocity of the pendulum at t = 0
- Now if we construct an abstract space with coordinates :math:`(\theta_{1},\theta_{2})`, the solutions :math:`(\theta_{1}(t),\theta_{2}(t))` would be moving along a line such as the one shown below. 

.. image:: https://raw.githubusercontent.com/SwaragThaikkandi/SMdRQA/main/docs/chapters/figs/pendulum_trajectories.svg
    :width: 800px
    :align: center

In short, this curve is called a **trajectory** and this abstract space is called **phase space** of the system. On the phase space each point can serve as an intial point. Motivation behind constructing phase space is to draw such trajectories and to extract information about the solutions. Becuase in many cases such geometric approach will give us solutions without solving the differential equations. 

For a general system:

.. math::

    \dot{x}_1 = f_1(x_1, x_2, x_3, \ldots, x_n)

    \dot{x}_2 = f_2(x_1, x_2, x_3, \ldots, x_n)

    \vdots

    \dot{x}_n = f_n(x_1, x_2, x_3, \ldots, x_n)

the phase space if an abstract space with coordinates :math:`(x_{1}, x_{2}, x_{n}, ..., x_{n})`, when the system is n-dimensional phase space, also reffered to as an **nth order system**


Nonlinear Problems are Hard
---------------------------

Nonlinear systems pose challenges for analytical solutions unlike linear systems. The key difference lies in how linear systems can be divided into manageable parts, solved separately, and then combined to find the overall solution. This approach simplifies complex problems and forms the basis for methods like normal modes, Laplace transforms, superposition arguments, and Fourier analysis. Linear systems are essentially the sum of their parts.

However, many natural phenomena don't follow this linear behavior. When different parts of a system interact, cooperate, or compete, nonlinear interactions occur. Everyday experiences often involve nonlinearities, where the principle of superposition doesn't apply straightforwardly. For instance, playing two favorite songs simultaneously doesn't double the enjoyment.

In physics, nonlinearity is crucial for phenomena like laser operation, turbulence formation in fluids, and the superconductivity of Josephson junctions.

In dealing with nonlinear systems, it's usually impossible to analytically determine the solutions. Even if explicit formulas exist, they're often too complex to offer meaningful insights. Instead, our focus is on understanding the qualitative behavior of the solutions. We aim to directly depict the system's phase portrait based on the characteristics of :math:`f(\mathbf{x})`.


Further discussion and details are outside the scope of this documentation, however, we would be happy to add specific topics if there is a pull request suggesting that might be useful


