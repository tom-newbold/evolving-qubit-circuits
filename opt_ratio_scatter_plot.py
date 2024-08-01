import os
import matplotlib.pyplot as plt
from pandas import read_csv
from linear_genetic_programming_utils import list_avr


def plot_scatters(filepath):
    os.makedirs(f'{filepath}/plots', exist_ok=True)

    with open(filepath+'/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        ITERATIONS = int(lines[0])
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')

    csv_to_plot = [f'{tp}_mult{m}.csv' for tp in test_params for m in multipliers]

    # compression ratio against unopt
    for metric in ['depth', 'length']:
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot] + [read_csv(filepath+f'/qiskit_mult{m}') for m in multipliers]):
            plt.clf()
            plt.title(f'{csv_to_plot[d_i][:-4].split("_")[0]} - {metric} ratios')
            min_max = [min(dataframe[f"r_unopt_{metric}"]), max(dataframe[f"r_unopt_{metric}"])]
            plt.plot(min_max, min_max, linestyle='dashed')
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"], s=10)
            plt.xlabel('$r_{unopt}$')
            if metric[0]=='d':
                plt.ylabel('$d_{opt}$')
            if metric[0]=='l':
                plt.ylabel('$len_{opt}$')
            plt.tight_layout()
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter.png')

    # all
    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'all {metric} ratios')
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            if d_i==0:
                min_max = [min(dataframe[f"r_unopt_{metric}"]), max(dataframe[f"r_unopt_{metric}"])]
                plt.plot(min_max, min_max, linestyle='dashed')
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"], s=2, label=csv_to_plot[d_i][:-4])
        plt.xlabel('$r_{unopt}$')
        if metric[0]=='d':
            plt.ylabel('$d_{opt}$')
        if metric[0]=='l':
            plt.ylabel('$len_{opt}$')
        plt.legend(loc='lower right', prop={'size': 'small'})
        plt.tight_layout()
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter.png')

    # scaled by runtime
    for metric in ['depth', 'length']:
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            plt.clf()
            plt.title(f'{csv_to_plot[d_i][:-4].split("_")[0]} - {metric} ratios (scaled by runtime)')
            #ratio = [a/b for a,b in zip(dataframe[f"r_unopt_{metric}"],dataframe[f"r_opt_{metric}"])]
            #scaled = [a/b for a,b in zip(ratio, dataframe["runtime"])]
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"]*dataframe["runtime"], s=10)
            plt.xlabel('$r_{unopt}$')
            if metric[0]=='d':
                plt.ylabel('$d_{opt}*runtime$')
            if metric[0]=='l':
                plt.ylabel('$len_{opt}*runtime$')
            plt.tight_layout()
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter_scaled.png')

    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'all {metric} ratios (scaled by runtime)')
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            #ratio = [a/b for a,b in zip(dataframe[f"r_unopt_{metric}"],dataframe[f"r_opt_{metric}"])]
            #scaled = [a/b for a,b in zip(ratio, dataframe["runtime"])]
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"]*dataframe["runtime"], s=2, label=csv_to_plot[d_i][:-4])
        plt.xlabel('$r_{unopt}$')
        if metric[0]=='d':
            plt.ylabel('$d_{opt}*runtime)')
        if metric[0]=='l':
            plt.ylabel('$len_{opt}*runtime$')
        plt.legend(loc='lower right', prop={'size': 'small'})
        plt.tight_layout()
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_scaled.png')

    """
    for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
        for metric in ['depth', 'length']:
            plt.clf()
            plt.title(f'{metric} ratios')
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"])
            plt.xlabel('r_unopt')
            plt.ylabel('r_opt')
            plt.tight_layout()
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter.png')


    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'{metric} ratios (scaled by runtime)')
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            r_opt_scaled = [a*b for a,b in zip(dataframe[f"r_unopt_{metric}"], dataframe["runtime"])]
            plt.scatter(dataframe[f"r_unopt_{metric}"], r_opt_scaled, label=csv_to_plot[d_i][:-4])
        
        plt.xlabel('r_unopt')
        plt.ylabel('r_opt * runtime')
        plt.legend(loc='upper left', prop={'size': 'small'})
        plt.tight_layout()
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter.png')

    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'{metric} compression ratios (scaled by runtime)')
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            r_opt_scaled = [1/(a*b) for a,b in zip(dataframe[f"r_unopt_{metric}"], dataframe["runtime"])]
            plt.scatter(dataframe[f"r_unopt_{metric}"], r_opt_scaled, label=csv_to_plot[d_i][:-4])
        
        plt.xlabel('r_unopt')
        plt.ylabel('(r_opt * runtime)$^{-1}$')
        plt.legend(loc='upper left', prop={'size': 'small'})
        plt.tight_layout()
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_inverted.png')

    '''
    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'{metric} ratios')
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_unopt_{metric}"], label=csv_to_plot[d_i][:-4])
        
        plt.xlabel('r_unopt')
        plt.ylabel('r_opt')
        plt.legend(loc='upper left', prop={'size': 'small'})
        plt.savefig(f'{filepath}/all_{metric}_ratio_scatter_pure.png')
    '''
    

    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'{metric} ratios')
        for d_i, dataframe in enumerate([read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]):
            grad = list_avr([1/r for r in dataframe[f"{metric}_compression_ratio"]])
            plt.plot([0, max(dataframe[f"r_unopt_{metric}"])], [0, max(dataframe[f"r_unopt_{metric}"])*grad], linestyle='dashed', label=csv_to_plot[d_i][:-4])
        
        plt.xlabel('r_unopt')
        plt.ylabel('r_opt')
        plt.legend(loc='upper left', prop={'size': 'small'})
        plt.tight_layout()
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_line.png')
    """


import sys

if __name__=="__main__":
    if len(sys.argv)>1:
        filepath = sys.argv[1]
    else:
        print('no filepath input, using prespecified')
        filepath = 'out/epsrc_qft3'

    plot_scatters(filepath)

    #out/epsrc_qft3_new