import numpy as np
from functools import reduce

#TODO: try binary weights and change distribution of weights positive versus negative

def magnetization(results):
    # get the average of the results array over the given time span
    return np.mean(results)

def correlation_length(results):
    pass

def thermal_average(results, t_span=None):
    # get the average of the results array over the given time span
    if t_span is None:
        t_span = (0, len(results))

    # results is a list of arrays, use reduce to sum them
    return reduce(lambda x, y: x + y, results[t_span[0]:t_span[1]]) / (t_span[1] - t_span[0])

def dfa(results, t_span=None):
    # get the detrended fluctuation analysis of the results array over the given time span
    if t_span is None:
        t_span = (0, len(results))
        
    