from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE
from linear_genetic_programming import Evolution, plot_many_averages
from grid_search import multiple_runs

from pandas import DataFrame


if __name__=="__main__":
    sets = {'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET}
    stats = {}
    to_plot = {}
    for set_name in sets:
        QFT_GEN = QFTGeneration(sets[set_name], 3)
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=8)

        to_plot[set_name], stats[set_name] = multiple_runs(E, iterations=20, plot=False)
    
    for set_name in sets:
        print(f'--{set_name}--')
        print(DataFrame.from_dict(stats[set_name]))
        plot_many_averages(to_plot[set_name], 'Generations', 'Circuit Fitness', legend=False)
