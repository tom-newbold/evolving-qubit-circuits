from qiskit.circuit.library import *
from qiskit import QuantumCircuit
from sklearn import linear_model
from skops.io import dump, load, get_untrusted_types
import pandas
import os

from linear_genetic_programming import AppliedProblemParameters, Genotype

#TRAINING_SET_SIZE = 10000
#VALIDATION_SET_SIZE = 100

os.makedirs('models', exist_ok=True)

CLIFFORD_T = [HGate(), XGate(), SGate(), SdgGate(), CXGate(), TGate(), TdgGate()]

class Redundancy:
    def __init__(self, training_sample_size=10000, validation_sample_size=100, gate_set=CLIFFORD_T, N=3):
        self.rand_gen = AppliedProblemParameters(gate_set, QuantumCircuit(N), genotype_len_bounds=(5,200), genotype_length_falloff='logarithmic')
        self.TRAINING_SET_SIZE = training_sample_size
        self.VALIDATION_SET_SIZE = validation_sample_size
        self.model = None

    def train(self, override=False):
        '''Generates random circuits and calculates their redundancy exactly,
           then uses linear regression to generate a formula to approximate'''
        if (not override) and self.model!=None:
            print('Coefficients already generated')
            return self.model.coef_
        df = pandas.DataFrame(columns=self.rand_gen.all_gate_combinations + ['total', 'redundancy'], dtype='Int64')
        # generate datapoints
        for i in range(self.TRAINING_SET_SIZE):
            if i%(self.TRAINING_SET_SIZE//50) == 0:
                progress = int(i/(self.TRAINING_SET_SIZE//50))
                print(f'training: {"#"*progress}{"-"*(50-progress)}', end='\r')
            rand_genotype = Genotype(self.rand_gen)
            distribution = Redundancy.count_gate_combinations(rand_genotype)
            redundancy = rand_genotype.remove_redundant_gates(approx=False)[1]
            distribution['redundancy'] = redundancy

            df.loc[i] = distribution
        print('')

        # train
        regr = linear_model.LinearRegression()
        regr.fit(df[self.rand_gen.all_gate_combinations].values, df['redundancy'])
        # SAVE DATAFRAME?
        self.model = regr

        return df, self.model.coef_
    
    def validate(self):
        '''Calulates max, min and average error across a smaller independent sample'''
        square_error = []
        for i in range(self.VALIDATION_SET_SIZE):
            if i%(self.VALIDATION_SET_SIZE//10) == 0:
                progress = int(i/(self.VALIDATION_SET_SIZE//10))
                print(f'validating: {"#"*progress}{"-"*(10-progress)}', end='\r')
            rand_genotype = Genotype(self.rand_gen)
            square_error.append(self.calc_error(rand_genotype))

        print('\nerrors')
        print(square_error)
        print(f'max: {max(square_error)}')
        print(f'min: {min(square_error)}')
        print(f'average: {sum(square_error)/len(square_error)}')
            
        return square_error

    @staticmethod
    def count_gate_combinations(genotype):
        '''Counts each independant gate permuation'''
        out = {gate:0 for gate in genotype.metadata.all_gate_combinations}
        i = 0
        j = 1
        while j<len(genotype.genotype_str):
            if genotype.genotype_str[i:j] in out:
                out[genotype.genotype_str[i:j]] += 1
                i = j
            j += 1

        out['total'] = sum([out[key] for key in out])
        return out
    
    def calc_error(self, genotype):
        '''Calculated difference of squares between exact and approximated values for reduncancy'''
        genotype_dict = Redundancy.count_gate_combinations(genotype)
        values = [genotype_dict[gate] for gate in genotype.metadata.all_gate_combinations]
        return (self.model.predict([values])[0] - genotype.remove_redundant_gates(approx=False)[1])**2
    
    def save_model(self, model_name):
        dump(self.model, f'models/{model_name}.skops')
        print('model saved')

    def load_model(self, model_name):
        try:
            loaded_model = load(f'models/{model_name}.skops')
            self.model = loaded_model
            print('model loaded')
            return self.model.coef_
        except:
            print(f'Cannot find model with name {model_name}.\nMake sure "{model_name}.skops" exists, or retrain the model.')
            return []



def analyse_gates(genotype):
    '''DEPRECATED'''
    distribution = dict(genotype.to_circuit().count_ops())
    distribution['total'] = sum([distribution[key] for key in distribution])
    reduced, redundancy = genotype.remove_redundant_gates()
    distribution['redundancy'] = redundancy
    return distribution


if __name__=="__main__":
    import matplotlib.pyplot as plt

    redundancy = Redundancy(N=3)
    
    target_filename = f'clifford-t-redundancy-model-{redundancy.TRAINING_SET_SIZE}samples'
    if len(redundancy.load_model(target_filename))==0:
        df, coef = redundancy.train()
        print(coef)

        plt.title('sample length range')
        plt.boxplot(df['total'].values)
        plt.ylim(0, 100)
        plt.show()

        redundancy.save_model(target_filename)
    
    
    square_error = redundancy.validate()

    plt.clf()
    plt.title('error')
    plt.boxplot(square_error)
    plt.ylim(0, max(square_error))
    plt.show()

    '''
    import time
    t_1 = 0
    t_2 = 0
    for _ in range(1000):
        rand_genotype = Genotype(app)
        start_time = time.time()
        rand_genotype.remove_redundant_gates()
        t_1 += time.time() - start_time
        start_time = time.time()
        genotype_dict = {gate:0 for gate in rand_genotype.metadata.all_gate_combinations}
        i = 0
        j = 1
        while j<len(rand_genotype.genotype_str):
            if rand_genotype.genotype_str[i:j] in genotype_dict:
                genotype_dict[rand_genotype.genotype_str[i:j]] += 1
                i = j
            j += 1
        values = [genotype_dict[gate] for gate in rand_genotype.metadata.all_gate_combinations]
        regr.predict([values])[0]
        t_2 += time.time() - start_time
    print(f'pure time: {t_1}')
    print(f'aproximation time: {t_2}')
    print(f'{t_1/t_2}x speedup')
    '''