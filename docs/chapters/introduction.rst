============
Introduction
============

RQA
----

Recurrent Quantification Analysis (RQA) has become a widely used tool for evaluating interpersonal synchrony in various studies. Its popularity stems from its minimal assumptions regarding the system's components and its robustness to dynamic variability in the signal, also known as non-stationarity (Marwan et al., 2007).

RQA operates on the principle that in a complex system governed by multiple interdependent components, any measured variable or dimension (along with its time-delayed copies) can be employed to reconstruct the system's dynamical behavior, represented as its phase space or state space (Takens, 1981). The phase space is a graphical depiction of all potential states of the system across time and comprises the coordinates or dimensions necessary to define a specific state of the system (Huffaker et al., 2017).

Quantifying recurrence, or the frequency with which a trajectory revisits a point in the phase space, is a fundamental aspect of RQA. This measure provides insights into the system's dynamics, revealing patterns of recurrence that reflect underlying processes and interactions.

MdRQA
-----

Multidimensional Recurrence Quantification Analysis (MdRQA) is an extension of RQA that enables the simultaneous consideration of time series data from more than two components or dimensions of the system for computing recurrence at the group level (Wallot et al., 2016; Gordon et al., 2021; Tomashin et al., 2022).

MdRQA expands the capabilities of traditional RQA by incorporating multidimensional data, which can include multiple variables, attributes, or features of the system. This extension allows researchers to capture complex interactions and dependencies across multiple dimensions, providing a more comprehensive analysis of the system's dynamics and behaviors.

In the following sections, we will explore the methodology, algorithms, and practical applications of MdRQA, and sliding window approach as well as the implementation of these algorithms as part of this package
