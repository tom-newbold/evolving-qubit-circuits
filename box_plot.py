import os
import matplotlib.pyplot as plt
from pandas import read_csv

from experiments import ALL_TESTS

def boxplot_from_folder(filepath="", fitness_reference=None):
    """takes folder of experimental results and plots box plots"""
    with open(filepath+'/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        ITERATIONS = int(lines[0])
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')

    for g in ["testparam","multiplier"]:
        os.makedirs(f'{filepath}/{g}_grouping', exist_ok=True)

    csv_to_plot = [[f'{tp}_mult{m}.csv' for tp in test_params for m in multipliers],[f'{tp}_mult{m}.csv' for m in multipliers for tp in test_params]]
    if len(multipliers)==1:
        csv_to_plot = [csv_to_plot[0]]
    columns_to_plot = [("peak_fitness",[0,1]),("generations_taken_to_converge",[0,50]),("runtime",[]),("peak_fitness/runtime",[]),("best_genotype_length",[0,40]),("best_genotype_depth",[0,10])]
    # extracts dataframes
    dataframe_orders = [[read_csv(filepath+'/'+csv_filename) for csv_filename in c] for c in csv_to_plot]

    for d_i, dataframes in enumerate(dataframe_orders):
        for c, r in columns_to_plot:
            if False in [c in d for d in dataframes] and '/' not in c:
                # checks that column exisits in every dataframe, this is skiped for compound keys (division)
                continue
            data = []
            labels = []
            for i, d in enumerate(dataframes):
                # for each row, add data and format column label
                if '/' in c:
                    c = c.split('/')
                    data.append(d[c[0]]/d[c[1]])
                    c = '/'.join(c)
                else:
                    data.append(d[c])
                csv_name = csv_to_plot[d_i][i].rstrip(".csv").split("_")
                labels.append(f'{csv_name[0]}\n{csv_name[1]}')
                #labels.append(f'{csv_name[0]}')
            plt.clf()
            #plt.figure().set_figwidth(3) # sets boxplot width
            plt.title(' '.join(c.split('_')))
            plt.boxplot(data, labels=labels, widths=[1.5/len(data) for _ in data])
            if fitness_reference and c=="peak_fitness":
                plt.axhline(fitness_reference, c='r', linewidth=0.5, linestyle='dashed')
            if len(r)!=0 and r[1] >= max([max(boxplot) for boxplot in data]):
                # set vertical limit/ticks if range is provided
                plt.ylim(r)
                plt.yticks([r[0] + i*(r[1]-r[0])/10 for i in range(11)])
            else:
                plt.ylim(bottom=0)
            #plt.xticks(rotation=45) # orient column labels
            plt.tight_layout() # refit to labels
            plt.grid(axis='y')
            plt.savefig(f'{filepath}/{["testparam","multiplier"][d_i]}_grouping/{c.replace("/","_")}_boxplot.png')
            #plt.show()

if __name__=="__main__":
    folder = '' # should end in slash, or be empty
    
    subfolders = [name for name in os.listdir('out/'+folder) if os.path.isdir('out/'+folder+name)]
    print(subfolders)
    for subdir in subfolders:
        if any(test_string in subdir for test_string in ALL_TESTS):
            boxplot_from_folder(f'out/{folder}{subdir}')
        else:
            print(f'{subdir} is not a test output folder')
    