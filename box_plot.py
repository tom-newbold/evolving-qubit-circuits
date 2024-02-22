from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

if __name__=="__main__":
    ITERATIONS = 10
    setnames = ['reduced','overcomplete']
    multipliers = [2,4,8]

    csv_to_plot = []
    for s in setnames:
        for m in multipliers:
            csv_to_plot.append(f"iter{ITERATIONS}_{s}_mult{m}.csv")
    #csv_to_plot = ["5iterations_reduced_mult2.csv","5iterations_reduced_mult4.csv",
    #               "5iterations_overcomplete_mult2.csv","5iterations_overcomplete_mult4.csv"]
    
    columns_to_plot = [("peak_fitness",[0,1]),("generations_taken_to_converge",[0,50])]
    dataframes = [read_csv('out/'+csv_filename) for csv_filename in csv_to_plot]

    STEPS = 10
    for c, r in columns_to_plot:
        data = []
        labels = []
        for i, d in enumerate(dataframes):
            data.append(d[c])
            csv_name = csv_to_plot[i].strip(".csv").split("_")
            labels.append(f'{csv_name[1]} x{csv_name[2][-1]}')
        plt.title(' '.join(c.split('_')))
        plt.ylim(r)
        plt.yticks([r[0] + i*(r[1]-r[0])/STEPS for i in range(STEPS+1)])
        plt.xticks(rotation=20)
        plt.boxplot(data, labels=labels)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()