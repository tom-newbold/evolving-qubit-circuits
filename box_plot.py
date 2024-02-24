from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
#TEST_PARAMETERS = [['random','stochastic','evolution'],
#                   ['reduced','overcomplete'],
#                   [f'{q}qubits' for q in [3,4,5]],
#                   [f'crossover{c}insertdelete{i}' for i in [1] for c in [3,4,5,6,7]]]#for x in ['single','double']

if __name__=="__main__":
    filepath = 'out/autosave_test_2'
    with open(filepath+'/params.txt','r') as file:
        lines = [l.strip('\n') for l in file.readlines()]
        ITERATIONS = int(lines[0])
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')
    #ITERATIONS = 10
    #multipliers = [5]#[2,4,8]

    csv_to_plot = [f'{tp}_mult{m}_boxplot.csv' for m in multipliers for tp in test_params]
    #for tp in test_params[3]:
    #    for m in multipliers:
    #        csv_to_plot.append(f"{tp}_mult{m}_boxplot.csv")
    #csv_to_plot = ["5iterations_reduced_mult2.csv","5iterations_reduced_mult4.csv",
    #               "5iterations_overcomplete_mult2.csv","5iterations_overcomplete_mult4.csv"]
    
    columns_to_plot = [("peak_fitness",[0,1]),("generations_taken_to_converge",[0,50]),("runtime",[]),("peak_fitness/runtime",[])]
    dataframes = [read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]

    STEPS = 10
    for c, r in columns_to_plot:
        if False in [c in d for d in dataframes] and '/' not in c:
            continue
        #if False in [c.split('/')[0] in d for d in dataframes]+[c.split('/')[1] in d for d in dataframes]:
        #    continue
        data = []
        labels = []
        for i, d in enumerate(dataframes):
            if '/' in c:
                c = c.split('/')
                data.append(d[c[0]]/d[c[1]])
                c = '/'.join(c)
            else:
                data.append(d[c])
            csv_name = csv_to_plot[i].strip(".csv").split("_")
            labels.append(f'{csv_name[0]} x{csv_name[1][-1]}')
        plt.clf()
        plt.title(' '.join(c.split('_')))
        if len(r)!=0:
            plt.ylim(r)
            plt.yticks([r[0] + i*(r[1]-r[0])/STEPS for i in range(STEPS+1)])
        plt.xticks(rotation=20)
        plt.boxplot(data, labels=labels)
        plt.grid(axis='y')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'{filepath}/{c.replace("/","_")}_boxplot.png')