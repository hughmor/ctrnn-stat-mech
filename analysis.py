import numpy as np
from functools import reduce

from simulation import activity, energy
from context import sim_context
#TODO: try binary weights and change distribution of weights positive versus negative


### Post-simulation analysis functions

def nearest_neighbors(lattice, i, j):
    N = len(lattice)
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    for n in neighbors:
        x,y = n
        if x < 0:
            x = N-1
        if x >= N:
            x = 0
        if y < 0:
            y = N-1
        if y >= N:
            y = 0
    return [(x, y) for x, y in neighbors if 0 <= x < N and 0 <= y < N]

def find_domains(lattice):
    lattice = np.tanh(lattice)
    N = len(lattice)
    visited = set()
    domains = []

    def dfs(i, j):
        stack = [(i, j)]
        size = 0

        while stack:
            current_i, current_j = stack.pop()

            if (current_i, current_j) not in visited:
                visited.add((current_i, current_j))
                size += 1

                for x, y in nearest_neighbors(lattice, current_i, current_j):
                    if lattice[current_i, current_j] * lattice[x, y] > 0:
                        stack.append((x, y))

        return size

    for i in range(N):
        for j in range(N):
            if (i, j) not in visited:
                domain_size = dfs(i, j)
                domains.append(domain_size)

    return domains


def autocorrelation(mag):
    pass

def autocovariance(mag):
    pass

def average_magnetization(activities):
    # activities should be a list of activities over time in equilibrium
    return np.mean(activities)

def mean_magnetization(activities):
    # activities should be a list of activities over time in equilibrium
    return np.abs(average_magnetization(activities))

def absolute_magnetization(activities): ##
    # activities should be a list of activities over time in equilibrium
    return average_magnetization(np.abs(activities))

def succeptability(activities):
    # absolute_activities should be returned by absolute_magnetization
    N = sim_context['num_points']
    T = sim_context['temperature']
    return np.var(np.abs(activities)) * N * N / T

def average_energy(energies):
    # energies should be a list of energies over time in equilibrium
    return np.mean(energies)

def heat_capacity(energies):
    # energies should be a list of energies over time in equilibrium
    N = sim_context['num_points']
    T = sim_context['temperature']
    return np.var(energies) * N * N / (T * T)


def periodic_distance(i, j):
    'get distance between point (i,j) and the origin, accounting for periodic boundary conditions'
    N = sim_context['num_points']
    x = min(abs(i), abs(N-i))
    y = min(abs(j), abs(N-j))
    return np.sqrt(x*x + y*y)


def correlation_function(results, N_samples):
    results = np.array(results)
    # choose the coordinate 0,0 as the reference point (it's arbitrary)
    # compute its mean over the last half of the simulation
    N_t = results.shape[0] 
    N_n = results.shape[1]
    assert N_n == results.shape[2]

    s0_t = results[N_t//2:, 0, 0]
    fluct_0 = s0_t - np.mean(s0_t)

    # take N_samples randomly chosen other points
    # compute their mean over the last half of the simulation
    x, y = np.random.randint(0, N_n, N_samples), np.random.randint(0, N_n, N_samples)
    si_t = [results[N_t//2:, i, j] for i, j in zip(x, y)]
    flucts_i = [s - np.mean(s) for s in si_t]

    # compute the correlation function
    corr_i = [fluct_0 * fluct_i for fluct_i in flucts_i]
    dynamic_correlation = np.mean(corr_i, axis=1) # axes are [coords, time, x, y] ?

    # compute the distance between the points and the origin
    dist_i = [periodic_distance(i,j) for i, j in zip(x, y)]

    return dist_i, dynamic_correlation


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
        
def simulation_steady(state, steady_state_window=20, steady_state_threshold=0.5):
    # noise_level = calculate_noise_level(state['grid'])
    # if noise_level >= max_noise_level: # above Tc
    #     return True

    #activity = state['activity']
    energy = state['energy']
    if len(energy) < steady_state_window:
        return False

    # recent_activities = activity[-steady_state_window:]
    # max_diff = np.max(recent_activities) - np.min(recent_activities)

    recent_energy = energy[-steady_state_window:]
    max_diff = np.max(recent_energy) - np.min(recent_energy)

    if max_diff <= steady_state_threshold:
        return True

    # smooth_m = np.convolve(activity, np.ones(steady_state_window) / steady_state_window, mode='valid')
    # mag_gradient = np.gradient(smooth_m)

    # if np.abs(np.mean(mag_gradient[-steady_state_window:])) <= steady_state_threshold:
    #     return True

    return False

def calculate_noise_level(states):
    differences = [np.abs(states[i + 1] - states[i]) for i in range(len(states) - 1)]
    noise_level = np.mean([np.std(diff) for diff in differences])
    return noise_level
