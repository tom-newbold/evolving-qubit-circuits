from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

if __name__=="__main__":
    ITERATIONS = 5
    multipliers = [2]#[2,4,8]

    csv_to_plot = []
    for test_param in ['random','stochastic','evolution']:#['reduced','overcomplete']:
        for m in multipliers:
            csv_to_plot.append(f"iter{ITERATIONS}_{test_param}_mult{m}.csv")
    #csv_to_plot = ["5iterations_reduced_mult2.csv","5iterations_reduced_mult4.csv",
    #               "5iterations_overcomplete_mult2.csv","5iterations_overcomplete_mult4.csv"]
    
    columns_to_plot = [("peak_fitness",[0,1]),("generations_taken_to_converge",[0,50]),("runtime",[0,10]),("peak_fitness/runtime",[])]
    dataframes = [read_csv('out/'+csv_filename) for csv_filename in csv_to_plot]

    STEPS = 10
    for c, r in columns_to_plot:
        print('/' in c)
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
            labels.append(f'{csv_name[1]} x{csv_name[2][-1]}')
        plt.title(' '.join(c.split('_')))
        if len(r)!=0:
            plt.ylim(r)
            plt.yticks([r[0] + i*(r[1]-r[0])/STEPS for i in range(STEPS+1)])
        plt.xticks(rotation=20)
        plt.boxplot(data, labels=labels)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()