import matplotlib.pyplot as plt
from pandas import read_csv

if __name__=="__main__":
    filepath = 'out/autosave_test_2'
    with open(filepath+'/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        ITERATIONS = int(lines[0])
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')

    csv_to_plot = [f'{tp}_mult{m}_boxplot.csv' for m in multipliers for tp in test_params]
    columns_to_plot = [("peak_fitness",[0,1]),("generations_taken_to_converge",[0,50]),("runtime",[]),("peak_fitness/runtime",[])]
    # extracts dataframes
    dataframes = [read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]


    for c, r in columns_to_plot:
        if False in [c in d for d in dataframes] and '/' not in c:
            # checks that column exisits in every dataframe, this is skiped for compound keys (division)
            continue
        #if False in [c.split('/')[0] in d for d in dataframes]+[c.split('/')[1] in d for d in dataframes]:
        #    continue
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
            csv_name = csv_to_plot[i].strip(".csv").split("_")
            labels.append(f'{csv_name[0]} x{csv_name[1][-1]}')
        plt.clf()
        plt.title(' '.join(c.split('_')))
        if len(r)!=0:
            # set vertical limit/ticks if range is provided
            plt.ylim(r)
            plt.yticks([r[0] + i*(r[1]-r[0])/10 for i in range(11)])
        plt.xticks(rotation=20) # orient column labels
        plt.boxplot(data, labels=labels)
        plt.grid(axis='y')
        plt.tight_layout() # fit to labels
        plt.savefig(f'{filepath}/{c.replace("/","_")}_boxplot.png')
        #plt.show()