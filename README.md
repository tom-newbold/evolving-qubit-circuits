# Evolutionary Algorithms for Generating Quantum Logic Gate Circuits

This software suite simplifies the process of applying a genetic programming methodology to quantum logic gate circuits. Provided alongside the base suite are examples of application to the Toffoli Gate problem and the Quantum Fourier Transform.

## Setup

- `pip install -r requirements.txt` will install the required dependencies for the project
- `python -m unittest -v` will run the unit test suite

## Using the evolutionary suite

### Gate Set
Make sure to first specify the set of gates you want to be usable by the generated circuits. Any of the base classes from `qiskit.circuit.library` can be used - create a list of instances of these classes (i.e. any gates with parameters must have these values specified, but any value can be used as it will be changed based on the value specified by the genotype). Symbols can be automatically generated, but equally they can be prespecified by using a dictionary rather than a list. *N.B. only a single symbol is permitted per gate; if a gate is indexed with multiple symbols, this will be overwritten by the first available auto-generated symbol.*

### Inputs
The fitness function requires a truth table to work on, which means a list of input states can be provided. These could either be the basis states for the system, or randomly generated intermediary states, though from testing it is recommended to use the former for better performance. If none are provided, the basis states for are taken for the number of qubits specified.

### Problem Parameters
Next, pass these into the constructor for `AppliedProblemParameters`, which uses only Mean Squared Fidelity as the fitness metric. If generated manually, a list of output states can be passed in; requires the number of qubits in the system to be specified. Alternatively, if a `QuantumCircuit` object (with the desired behaviour to be replicated) is available, this can be used instead, and the output states will be determined internally, along with the number of qubits. It is recommended to specify values for genotype generation (`min_length`, `max_length`, `falloff`) in `ProblemParameters` rather than with each call to `evolutionary_search()`.

### Evolution
This object is used to create an instance of `Evolution`. Default values are specified for all search parameters, but these can also be overwritten to customise the environment. Calling `random_search()`, `stochastic_hill_climb()` or `evolutionary_search()` runs the chosen algorithm. Again, each of these functions has a range of parameters, all with default values, which influence the random generation and evolution. Returned is a sorted list of `Genotype`s - the final generation - and the fitnesses of the top sample at every generation (which can be plotted with the provided functions).

### Bulk Runs

`bulk_runs.py` contains functions which allow bulk runs - graphs showing average sample fitness across generations per run can be plotted. `qft_experiments.py` and `toffoli_experiments.py` show the algorithm run with different parameters for statistical analysis.
