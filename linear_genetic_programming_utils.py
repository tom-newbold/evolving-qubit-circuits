import math
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector

def encode_to_letter(n):
    '''26 (english) capitals , 26 (english) lower case,
       10 valid (greek) upper case, 18 valid (greek) lower case
       80 allowable symbols'''
    if n < 0:
        return None
    if n < 26:
        key = chr(ord('A')+n)
    elif n < 52:
        key = chr(ord('a')+n-26)
    elif n < 62:
        key = ['Γ','Δ','Θ','Λ','Ξ','Π','Σ','Φ','Ψ','Ω'][n-52]
    elif n < 80:
        key = ['α','β','γ','δ','ε','ζ','η','θ','λ','μ',
               'ξ','ρ','σ','τ','φ','χ','ψ','ω'][n-62]
    else:
        return None
    return key

def basis_states(N=3):
    """returns a list of the 2**N basis states for an N-qubit system"""
    return [Statevector.from_int(i, 2**N) for i in range(2**N)]

def ansi(n=0):
    '''returns the ANSI escape code for n (used for text colouring)'''
    try:
        n = int(n)
        if n>=0 and n<10:
            return f'\033[0{str(n)}m'
        elif n>=10 and n<100:
            return f'\033[{str(n)}m'
        else:
            return ''
    except:
        return ''
    
def list_avr(l):
    """calculates the average value of a single list"""
    return sum(l)/len(l)

def get_averages_list(float_list):
    """calculate the average value across each list at every index
       float_list is a expected to be a list of lists"""
    if type(float_list)!=list:
        return None
    if type(float_list[0])!=list:
        return None
    return [list_avr([y[i] for y in float_list]) for i in range(len(float_list[0]))]

def get_max_list(float_list):
    """calculate the maximum value across each list at every index
       float_list is a expected to be a list of lists"""
    if type(float_list)!=list:
        return None
    if type(float_list[0])!=list:
        return None
    return [max([y[i] for y in float_list]) for i in range(len(float_list[0]))]

def get_min_list(float_list):
    """calculate the minimum value across each list at every index
       float_list is a expected to be a list of lists"""
    if type(float_list)!=list:
        return None
    if type(float_list[0])!=list:
        return None
    return [min([y[i] for y in float_list]) for i in range(len(float_list[0]))]

def smooth_line(float_list, half_width=2):
    """calculate a (2*half_width+1) point moving average to smooth float_list"""
    half_width = max(half_width,1) # prevents zero/negative
    if type(float_list)!=list:
        return None
    if len(float_list)<=2*half_width:
        return float_list
    out = []
    for i in range(len(float_list)):
        a = i-half_width
        if a < 0: a=0
        b = i+1+half_width
        if b > len(float_list): b = len(float_list)
        out.append(list_avr(float_list[a:b]))
    return out

def plot_list(float_list, x_label=None, y_label=None, plot_average=True):
    """plots a list of floats"""
    plt.close()
    plt.figure(layout='constrained')
    if type(float_list[0])==list:
        x_axis = [i for i in range(len(float_list[0]))]
        if plot_average:
            plt.plot(x_axis, get_max_list(float_list), linewidth=20/(20+len(float_list)), label='best (overall)')
            plt.plot(x_axis, get_min_list(float_list), linewidth=20/(20+len(float_list)), label='worst (in sample)')
            plt.plot(x_axis, get_averages_list(float_list), linestyle='dashed', label='average (of sample)')
            plt.legend(loc='upper left', prop={'size': 'small'})
        else:
            for j in range(len(float_list)):
                plt.plot(x_axis, float_list[-(j+1)], linewidth=20/(20+len(float_list)))
    else:
        x_axis = [i+1 for i in range(len(float_list))]
        plt.plot(x_axis, float_list)
    
    while len(x_axis) > 20:
        x_axis = [x_axis[i*5] for i in range(len(x_axis)//5+1)]
    #plt.xticks([0]+x_axis)
    plt.xticks(x_axis)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    try:
        max_value = max(1, max(float_list))
    except:
        max_value = max([max(float_list[i]) for i in range(len(float_list))]+[1])
    if max_value > 1:
        max_value = math.ceil(max_value/10)*10
        plt.yticks([x*10 for x in range(max_value//10+1)])
    else:
        plt.yticks([x/10 for x in range(1+math.ceil(10*max_value))])
    plt.xlim([x_axis[0],x_axis[-1]])
    plt.ylim([0,max_value])
    
    plt.grid()

def plot_many_averages(float_lists, x_label=None, y_label=None, plot_trendline=True, trendline_halfwidth=4, legend=True, reference_line=None):
    plt.close()
    plt.figure(layout='constrained')
    lw = 20/(20+len(float_lists[0]))

    x_axis = [i for i in range(len(float_lists[0][0]))]
    max_values = [1] + [max(get_max_list(float_list)) for float_list in float_lists]
    to_plot = [get_averages_list(float_list) for float_list in float_lists]

    if reference_line!=None:
        plt.axhline(reference_line, c='r', linestyle='dashed', label='ideal line')

    if plot_trendline:
        trend = smooth_line(get_averages_list(to_plot), half_width=trendline_halfwidth)
        plt.plot(x_axis[trendline_halfwidth:], trend[trendline_halfwidth:], linewidth=1.25*lw, label='trendline', color='black')
    
    for run, line in enumerate(to_plot):
        plt.plot(x_axis, line, linewidth=lw, linestyle='dashed', label=f'run {run+1}')
    if legend:
        plt.legend(loc='upper left', ncols=math.ceil(len(float_lists)/5), prop={'size': 'small'})

    max_value = max(max_values)
    if max_value > 1:
        max_value = math.ceil(max_value/10)*10
        plt.yticks([x*10 for x in range(max_value//10+1)])
    else:
        plt.yticks([x/10 for x in range(1+math.ceil(10*max_value))])

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.xlim([x_axis[0],x_axis[-1]])
    plt.ylim([0,max_value])

    plt.grid()


def remove_duplicates(genotype_list):
    '''removes any genotypes with duplicate string representations.
       removing duplicates by circuit representation would produce a much larger overhead
       N.B. this function is likely not the most efficient way'''
    seen_genotypes = []
    out = []
    for i in range(len(genotype_list)):
        if genotype_list[i].genotype_str not in seen_genotypes:
            seen_genotypes.append(genotype_list[i].genotype_str)
            out.append(genotype_list[i])
    return out