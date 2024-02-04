# Simulating traffic flow using the Nagel-Schreckenberg model

Brief description of the project: In this project we want to study traffic flow and how congestion emerges. For this, we use the Nagel Schreckenberg cellular automota model to observe the effect of different parameters such car density, max velocity and driver behaviour. In particular, we wish to study the occurence of phase transitions when varying the values of these parameters.


## Background and motivation

With cars being by far the most used form of transport worldwide, it is crucial to understand the dynamics of road traffic systems and how congestion emerges in these systems. Despite traffic being dependent on many individually complex drivers, the fundamental dynamics of a road as a system can be actually be modelled using basic methods such as cellular automata. Among such CA, the simplest non-deterministic model of single-lane traffic is the Nagel-Schreckenberg model. Using this model, this project aims to conduct analysis on the emergent behaviour of traffic jams with respect to particular parameters to better understand the important factors of such a traffic system.


## Research question

How does varying different parameter values affect the Nagel-Schreckenberg model?

### Sub research questions

Under which initial condition of density do we see congestion emerge?
Under which initial conditions of car speed do we see congestion emerge?
How does erratic driving behaviour, i.e. sharp braking and accelerating, affect the system?


## Hypotheses
We hypothesize that there exists a critical density threshold below which traffic flows smoothly, but beyond which congestion rapidly ensues. Furthermore, we expect there to be similar critical values for other parameters of the model.


## Installation
To install the necessary packages, run the command
`pip install -r requirements.txt`


## References

Nagel, Kai & Schreckenberg, Michael. (1992). A cellular automaton model for freeway traffic. Journal de Physique I. 2. 2221. 10.1051/jp1:1992277. 

Rickert, Marcus, et al. "Two lane traffic simulations using cellular automata." Physica A: Statistical Mechanics and its Applications 231.4 (1996): 534-550.