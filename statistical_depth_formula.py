import os
import pandas

from qiskit import QuantumCircuit
from linear_genetic_programming import Genotype, AppliedProblemParameters, CLIFFORD_T
from sklearn import linear_model, neural_network
from skops.io import load, dump


os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

class Depth_Model:
    def __init__(self, training_sample_size=10000, validation_sample_size=100, gate_set=CLIFFORD_T, N=3):
        self.rand_gen = AppliedProblemParameters(gate_set, QuantumCircuit(N), genotype_len_bounds=(5,200), genotype_length_falloff='logarithmic')
        self.TRAINING_SET_SIZE = training_sample_size
        self.VALIDATION_SET_SIZE = validation_sample_size
        self.model = None

    def train(self, override=False, input_df=None):
        '''Generates random circuits and calculates their depth exactly,
           then uses linear regression to generate a formula to approximate'''
        if (not override) and self.model!=None:
            print('Coefficients already generated')
            return self.model.coef_
        if input_df==None:
            try:
                pandas.read_csv(f'data/{self.rand_gen.qubit_count}qubit-depth-datapoints')
            except:
                df = pandas.DataFrame(columns=self.rand_gen.all_gate_combinations + ['total', 'depth'], dtype='Int64')
                # generate datapoints
                for i in range(self.TRAINING_SET_SIZE):
                    if i%(self.TRAINING_SET_SIZE//50) == 0:
                        progress = int(i/(self.TRAINING_SET_SIZE//50))
                        print(f'generating datapoints: {"#"*progress}{"-"*(50-progress)}', end='\r')
                    rand_genotype = Genotype(self.rand_gen)
                    distribution = Depth_Model.count_gate_combinations(rand_genotype)
                    distribution['depth'] = rand_genotype.to_circuit().depth()

                    df.loc[i] = distribution
                print('')
                df.to_csv(f'data/{self.rand_gen.qubit_count}qubit-datapoints')
        else:
            df = input_df

        # train
        regr = linear_model.LinearRegression()
        #regr = neural_network.MLPRegressor(hidden_layer_sizes=(2*len(df[self.rand_gen.all_gate_combinations].values)), alpha=0.1)
        regr.fit(df[self.rand_gen.all_gate_combinations].values, df['depth'])
        self.model = regr

        return df, [None]# self.model.coef_
    
    def validate(self):
        '''Calulates max, min and average error across a smaller independent sample'''
        square_error = []
        points = {'pure':[], 'predicted':[]}
        for i in range(self.VALIDATION_SET_SIZE):
            if i%(self.VALIDATION_SET_SIZE//10) == 0:
                progress = int(i/(self.VALIDATION_SET_SIZE//10))
                print(f'validating: {"#"*progress}{"-"*(10-progress)}', end='\r')
            rand_genotype = Genotype(self.rand_gen)
            square_error.append(self.calc_error(rand_genotype))
            
            points['pure'].append(rand_genotype.to_circuit().depth())
            points['predicted'].append(self.estimate_depth(rand_genotype))

        print('\nerrors')
        print(square_error)
        print(f'max: {max(square_error)}')
        print(f'min: {min(square_error)}')
        print(f'average: {sum(square_error)/len(square_error)}')
            
        return square_error, points

    @staticmethod
    def count_gate_combinations(genotype, calc_total=True):
        '''Counts each independant gate permuation'''
        out = {gate:0 for gate in genotype.metadata.all_gate_combinations}
        i = 0
        j = 1
        while j<len(genotype.genotype_str):
            if genotype.genotype_str[i:j] in out:
                out[genotype.genotype_str[i:j]] += 1
                i = j
            j += 1

        if calc_total: out['total'] = sum([out[key] for key in out])
        #---
        #distribution = dict(genotype.to_circuit().count_ops())
        #distribution['total'] = sum([distribution[key] for key in distribution])
        #return distribution
        return out
    
    def estimate_depth(self, genotype):
        '''Predicts the depth using the model'''
        genotype_dict = Depth_Model.count_gate_combinations(genotype, False)
        values = [genotype_dict[gate] for gate in genotype.metadata.all_gate_combinations]
        return self.model.predict([values])[0]

    def calc_error(self, genotype):
        '''Calculated difference of squares between exact and approximated values for depth'''
        return (self.estimate_depth(genotype) - genotype.to_circuit().depth())**2

    
    def save_model(self, model_name):
        dump(self.model, f'models/{model_name}.skops')
        print('model saved')

    def load_model(self, model_name):
        try:
            loaded_model = load(f'models/{model_name}.skops', ['sklearn.neural_network._stochastic_optimizers.AdamOptimizer'])
            self.model = loaded_model
            print('model loaded')
            return [None] #self.model.coef_
        except:
            print(f'Cannot find model with name {model_name}.\nMake sure "{model_name}.skops" exists, or retrain the model.')
            return []

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from quantum_fourier_transform import GATE_SET
    depthmodel = Depth_Model()#gate_set=GATE_SET)
    
    target_filename = f'depth-model-{depthmodel.rand_gen.qubit_count}'
    if len(depthmodel.load_model(target_filename))==0:
        df, coef = depthmodel.train()
        print(coef)

        '''
        plt.title('sample length range')
        plt.boxplot(df['total'].values)
        plt.ylim(0, 100)
        plt.show()
        '''

        depthmodel.save_model(target_filename)

        #plt.clf()
        #plt.plot(depthmodel.model.loss_curve_)
        #plt.show()
    
    
    square_error, points = depthmodel.validate()
    avr = sum(points['pure'])/len(points['pure'])
    ss_tot = sum([(p-avr)**2 for p in points["pure"]])
    print(f'r2: {1 - sum(square_error)/ss_tot}')

    plt.clf()
    plt.title('error')
    plt.boxplot(square_error)
    plt.ylim(0, max(square_error))
    plt.show()

    plt.clf()
    plt.title('error')
    line = [min(0, max(min(points['pure']), min(points['predicted']))),
            max(1, min(max(points['pure']), max(points['predicted'])))]
    plt.plot(line, line)
    plt.scatter(points['pure'], points['predicted'])
    plt.xlabel('pure')
    plt.ylabel('predicted')
    plt.show()

    # testing individual variable relations
    import pandas
    import numpy as np
    from linear_genetic_programming import Genotype
    df = pandas.DataFrame(columns=depthmodel.rand_gen.all_gate_combinations + ['total', 'depth', 'error', 'fitness'], dtype='Int64')
    error = []
    # generate datapoints
    for i in range(100):
        rand_genotype = Genotype(depthmodel.rand_gen)
        distribution = Depth_Model.count_gate_combinations(rand_genotype)
        distribution['depth'] = rand_genotype.to_circuit().depth()
        distribution['fitness'] = rand_genotype.get_fitness()
        distribution['error'] = depthmodel.calc_error(rand_genotype)

        df.loc[i] = distribution
        
    plt.clf()
    plt.scatter(df['fitness'], df['error'])
    plt.xlabel('fitness')
    plt.ylabel('depth error')
    plt.show()

    '''
    for var in redundancy.rand_gen.all_gate_combinations:
        plt.clf()
        plt.title(var)
        plt.scatter(df[var], df['redundancy'])
        x = np.linspace(0, int(max(df[var])), 2)
        y = redundancy.model.coef_[redundancy.rand_gen.all_gate_combinations.index(var)] * x
        plt.plot(x, y)
        plt.ylabel('redundancy')
        plt.show()
    '''
    
    import time
    t_1 = 0
    t_2 = 0
    for _ in range(1000):
        rand_genotype = Genotype(depthmodel.rand_gen)
        start_time = time.time()
        rand_genotype.to_circuit().depth()
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
        depthmodel.model.predict([values])[0]
        t_2 += time.time() - start_time
    print(f'pure time: {t_1}')
    print(f'aproximation time: {t_2}')
    print(f'{t_1/t_2}x speedup')