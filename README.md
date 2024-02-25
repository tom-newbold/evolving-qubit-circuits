# Evolutionary Algorithms for Generating Quantum Logic Gate Circuits

Quantum Computing is becoming an enticing area of study, on account of its hypothetical potential for revolutionising the field of computability. Quantum gate circuits utilise quantum phenomena to extend the classical logic gate model, making it exponentially more powerful. Yet, such circuits are significantly harder to comprehend and hence develop. Within the literature, genetic approaches are popular choices for overcoming this problem, having been shown to be very effective to this end.

The goal of this project is to explore quantum logic gate circuits for a range of problems, testing the applicability of genetic programming, and ultimately producing a general software suite for doing this.


## Using the evolutionary suite

### Gate Set
Make sure to first specify the set of gates you want to be usable by the generated circuits. Any of the base classes from `qiskit.circuit.library` can be used - create a list of instances of these classes (i.e. any gates with parameters must have these values specified, but any value can be used as it will be changed based on the value specified by the genotype). Symbols can be automatically generated, but equally they can be prespecified by using a dictionary rather than a list. *N.B. only a single symbol is permitted per gate; if a gate is indexed with multiple symbols, this will be overwritten by the first available auto-generated symbol.*

### Inputs
The fitness function requires a truth table to work on, which means a list of input states can be provided. These could either be the basis states for the system, or randomly generated intermediary states, though from testing it is recommended to use the former for better performance. If none are provided, the basis states for are taken for the number of qubits specified.

### Problem Parameters
Next, pass these into the constructor for one of the following classes:

`AppliedProblemParameters`: Uses only Mean Squared Fidelity as the fitness metric. If generated manually, a list of output states can be passed in; requires the number of qubits in the system to be specified. Alternatively, if a `QuantumCircuit` object (with the desired behaviour to be replicated) is available, this can be used instead, and the output states will be determined internally, along with the number of qubits.

`ProblemParametersCombined`: Uses both MSF and a Matrix Difference fitness combined. This requires a `QuantumCircuit` object be input, as its matrix representation is used to determine genotype fitness. An optional parameter for the *tolerance* of the MDF can be provided. *An optimal value for this has yet to be determined.*

### Evolution
This object is used to create an instance of `Evolution`. Default values are specified for all search parameters, but these can also be overwritten to customise the environment. Calling `random_search`, `stochastic_hill_climb` or `evolutionary_search` runs the chosen algorithm. Again, each of these functions has a range of parameters, all with default values, which influence the random generation and evolution. Returned is a sorted list of `Genotype`s - the final generation.

### Bulk Runs

`grid_search.py` contains functions which allow bulk runs - graphs showing average sample fitness across generations per run can be plotted. `qft_experiments.py` and `box_plot.py` show the algorithm run with different parameters for statistical analysis.