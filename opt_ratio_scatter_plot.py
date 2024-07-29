import os
import matplotlib.pyplot as plt
from pandas import read_csv
from linear_genetic_programming_utils import list_avr

filepath = 'out\eprsc_optimisers'

with open(filepath+'/params.txt','r') as file:
    # fetches run parameters in order to consruct csv filenames
    lines = [l.strip('\n') for l in file.readlines()]
    ITERATIONS = int(lines[0])
    multipliers = [int(m) for m in lines[1].split(',')]
    test_params = lines[2].split(',')

csv_to_plot = [f'{tp}_mult{m}.csv' for tp in test_params for m in [6]]
for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'{metric} ratios')
        plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"])
        plt.savefig(f'{filepath}/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter.png')


for metric in ['depth', 'length']:
    plt.clf()
    plt.title(f'{metric} ratios')
    for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
        r_unopt_scaled = [a/b for a,b in zip(dataframe[f"r_unopt_{metric}"], dataframe["runtime"])]
        r_opt_scaled = [a/b for a,b in zip(dataframe[f"r_unopt_{metric}"], dataframe["runtime"])]
        plt.scatter(r_unopt_scaled, r_opt_scaled, label=csv_to_plot[d_i][:-4])
        grad = list_avr([1/r for r in dataframe[f"{metric}_compression_ratio"]])
        plt.plot([0, 5], [0, 5*grad]) # TODO Scale correctly
    plt.legend() # ???
    plt.savefig(f'{filepath}/all_{metric}_ratio_scatter.png')
