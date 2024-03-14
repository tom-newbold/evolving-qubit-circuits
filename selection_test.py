import random
from linear_genetic_programming import Evolution
from quantum_fourier_transform import QFTGeneration, GATE_SET
from bulk_runs import multiple_runs
from box_plot import boxplot_from_folder
from experiments import Experiments

class TournamentEvolution(Evolution):
    def top_by_fitness(self, population, min_fitness=0, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True):
        """finds the best circuits in the population; top sample taken as well as uniform selection of remaining circuits"""
        by_fitness = Evolution.sort_by_fitness(population, min_fitness, prefer_short_circuits, prefer_long_circuits, remove_dupe)
        out = by_fitness[:self.SAMPLE_SIZE]
        for t in range(self.GENERATION_SIZE-self.SAMPLE_SIZE):
            try:
                tournament = random.choices(by_fitness[self.SAMPLE_SIZE:], k=self.SAMPLE_SIZE) # k is tournament size
                out.append(Evolution.sort_by_fitness(tournament, remove_dupe=True)[0])
            except:
                print(by_fitness[self.SAMPLE_SIZE:])
                break
        return out
    
class FitnessProportionateEvolution(Evolution):
    def top_by_fitness(self, population, min_fitness=0, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True):
        """finds the best circuits in the population; top sample taken as well as uniform selection of remaining circuits"""
        by_fitness = Evolution.sort_by_fitness(population, min_fitness, prefer_short_circuits, prefer_long_circuits, remove_dupe)
        out = by_fitness[:self.SAMPLE_SIZE]
        remaining = by_fitness[self.SAMPLE_SIZE:]
        for t in range(self.GENERATION_SIZE-self.SAMPLE_SIZE):
            try:
                next_g = random.choices(remaining, weights=[g.get_fitness() for g in remaining], k=1)[0]
                out.append(next_g)
                remaining.remove(next_g)
            except:
                break
        return out

ITERATIONS = 25
GENERATIONS = 50
if __name__=="__main__":
    N, M = 3, 6
    folder = 'out/selection_test'

    QFT_GEN = QFTGeneration(GATE_SET, N)
    stats = {}
    to_plot = {}

    test_methods = [('uniform', Evolution), ('tournament', TournamentEvolution), ('fitness', FitnessProportionateEvolution)]

    for param, method in test_methods:
        to_plot[param], stats[param] = multiple_runs(method(QFT_GEN, gen_mulpilier=M, number_of_generations=GENERATIONS, sample_percentage=0.1),
                                                     iterations=ITERATIONS, plot=False, save_dir=folder)
    #to_plot['tournament'], stats['tournament'] = multiple_runs(TournamentEvolution(QFT_GEN, gen_mulpilier=M, number_of_generations=GENERATIONS, sample_percentage=0.1),
    #                                                           iterations=ITERATIONS, plot=False, save_dir='out/tournament_selection_test/')
    
    e_instance = Experiments(QFT_GEN,ITERATIONS,[M],folder)
    # plot fitness graphs
    for param, method in test_methods:
        e_instance.output(to_plot, stats, param, M)
    # save params
    with open(folder+'/params.txt','w') as file:
        file.write(f'{ITERATIONS}\n6\n{",".join(list(stats.keys()))}')
        file.close()
    # plot boxplots
    boxplot_from_folder(folder,(2**N-1)/(2**N))