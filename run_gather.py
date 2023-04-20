#%%

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from tqdm import tqdm
import argparse
import json
from functools import partial
import sys
from util import check_disk_space_prompt_user


def create_grid(grid_size_x, grid_size_y, initial=None):
    # Define the grid size and the step size
    nx, ny = grid_size_x, grid_size_y
    dx, dy = 1, 1

    # Create the grid using numpy functions
    x = np.linspace(0, nx * dx, nx, endpoint=False)
    y = np.linspace(0, ny * dy, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Define the initial conditions for the grid
    if initial is None or initial == 'uniform':
        initial_conditions = 2*np.random.rand(nx, ny) - 1
    elif initial == 'gaussian':
        initial_conditions = np.random.normal(0, 1, (nx, ny))
    elif initial == 'zero':
        initial_conditions = np.zeros((nx, ny))
    else:
        raise ValueError(f"invalid grid initial condition: {initial}")

    return initial_conditions

def create_biases(grid, initial=None):
    biases = np.zeros_like(grid)

    if initial is None or initial == 'uniform':
        # fill the grid with random values from a uniform distribution
        biases = 2*np.random.rand(biases.shape[0], biases.shape[1]) - 1
    if initial == 'random':
        # fill the grid with random values from a normal distribution
        biases = np.random.normal(0, 1, biases.shape)
    elif initial == 'zero':
        biases = np.zeros_like(biases)
    else:
        raise ValueError(f"invalid bias initial condition: {initial}")
    
    return biases

def wrapped_index(array, i, j):
    nx, ny = array.shape
    i_wrapped = i % nx
    j_wrapped = j % ny
    return array[i_wrapped, j_wrapped]

# 1. Divide the grid into smaller chunks with overlaps
def divide_grid(grid, num_chunks, overlap=1):
    # Divide the grid into num_chunks along each axis
    nx, ny = grid.shape
    cx, cy = num_chunks
    grid_chunks = []

    chunk_size_x = nx // cx
    chunk_size_y = ny // cy

    for i in range(cx):
        for j in range(cy):
            x_start = max(0, i * chunk_size_x - overlap)
            x_end = min(nx, (i + 1) * chunk_size_x + overlap)
            y_start = max(0, j * chunk_size_y - overlap)
            y_end = min(ny, (j + 1) * chunk_size_y + overlap)
            grid_chunks.append(grid[x_start:x_end, y_start:y_end])

    return grid_chunks

def ds_dt(t, s, inp):
    return -s + inp

def do_weighting(s, coupling):
    # s is the state of the grid (NxN)
    # coupling is the coupling matrix (MxM) M < N
    
    # convolve s with the coupling matrix and return result with same shape as s
    return convolve2d(s, coupling, mode='same', boundary='wrap')
    
def coupled_diff_eqs(t, y, params):
    inp = params[0]
    return ds_dt(t, y, inp)
    
# 2. Define a function to run the simulation for a single chunk
def run_simulation(chunk, params, t_span):

    # params contains bias, coupling matrix, random process
    coupling, noise_process, bias = params # TODO: this might bug based on the way params are defined in parallel_simulation
    noise = np.zeros_like(chunk)
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            noise[i,j] = noise_process(0.0) # TODO: pulled this from above - should give it a correct t value
    inp = bias + noise + do_weighting(np.tanh(chunk), coupling)

    # Flatten the chunk to a 1D array
    oned_chunk = chunk.ravel()
    oned_inp = inp.ravel()
    solve_ivp_params = (oned_inp,)

    # Run the simulation for the given chunk
    sol = solve_ivp(coupled_diff_eqs, t_span, oned_chunk, args=(solve_ivp_params,), method='RK45')
    ret_1d = sol.y

    # Reshape the solution to the original shape of the chunk
    # ret_1d[:,0] is the initial condition
    # ret_1d[:,-1] is the final condition
    # TODO: right now, not saving the intermediate time steps
    ret = ret_1d[:,-1].reshape(chunk.shape)
    return ret

# 3. Set up multiprocessing and run the simulations in parallel
def parallel_simulation(initial_conditions, biases, params, t_span, num_chunks):
    grid_chunks = divide_grid(initial_conditions, num_chunks)
    bias_chunks = divide_grid(biases, num_chunks)

    # Create a list of parameters for each thread
    # add the bias chunk to the params
    thread_params = [params + (bias_chunk,) for bias_chunk in bias_chunks]

    # Run the simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_simulation, grid_chunks, thread_params, [t_span] * len(grid_chunks)))

    return results

# 4. Combine the chunks and save the state of the whole grid at every time step
def combine_chunks_and_save(results, params):
    # Combine the chunks and save the state of the whole grid at every time step
    # results is a list of arrays returned from the parallel simulations

    # get the shape of the grid
    nx, ny = results[0].shape

    # get the number of time steps
    num_time_steps = results[0].shape[1] #TODO: this might bug out if results take a different number of intermediate time steps

    # create an empty array to store the state of the whole grid at every time step
    grid_state = np.zeros((nx, ny, num_time_steps))

    # loop over the results
    for i in range(len(results)): 
        # get the chunk
        chunk = results[i]
        # get the shape of the chunk
        cx, cy = chunk.shape
        # get the starting and ending indices of the chunk in the grid
        x_start = i * cx
        x_end = (i + 1) * cx
        y_start = i * cy
        y_end = (i + 1) * cy
        # store the chunk in the grid
        grid_state[x_start:x_end, y_start:y_end, :] = chunk

    # save the state of the whole grid at every time step
    np.save('grid_state.npy', grid_state)

def serial_simulation(initial_conditions, biases, params, t_span):
    # Run simulation without parallelization
    params = params + (biases,)
    sol = run_simulation(initial_conditions, params, t_span)
    return sol

def create_coupling_matrix(N):
    # create an NxN matrix with gaussian distribution from the center
    matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            matrix[i,j] = np.exp(-((i-N//2)**2 + (j-N//2)**2)/2)
    return matrix



def execute_simulation(args):
    print("setting up grid and simulation...")
    # set up grid
    N = M = args.num_points
    #grid = create_grid(N, M)
    # set up biases according to some random distribution
    
    # this is unhinged but fun lol
    biases = create_biases(grid:=create_grid(N, M))
    # actually should I do this backwards? create the biases first and then create the grid based on the bias; this will allow the grid to initialize to the biases
    
    # set up time parameters
    num_eval = args.num_eval
    dt = args.timestep
    t_span = (0.0, num_eval * dt)
    t_eval = np.linspace(t_span[0], t_span[1], num_eval)

    # set up coupling matrix
    coupling_radius = 3
    coupling = create_coupling_matrix(coupling_radius)

    # set up random process
    noise = 0.01
    noise_process = lambda t: np.random.normal(0, noise)

    params = (coupling, noise_process)
    # run simulation over a number of timesteps
    results = []

    check_disk_space_prompt_user(grid, t_eval)

    # use tqdm to show progress bar
    for t in tqdm(t_eval, desc="     "):
        res = serial_simulation(grid, biases, params, (t, t+1))
        results.append(deepcopy(res))
        grid = res

    print("finished! saving results...")
    # save the results
    file = args.output # 'grid_randombias_0_01_noise'
    np.save(file, results)

    with open(f"{args.output.name}.metadata", "w") as f_metadata:
        json.dump(vars(args), f_metadata, default=repr)

    print("done!")

#%%

"""
    TODO List
    first, accept these parameters as command line arguments:
    - noise process / values
    - dtype
    - type of coupling matrix
        - random
        - uniform
        - gaussian
    - parameters of matrix
        - radius

    improvements:
    - calculate required memory for final arrays, intermediate chunks, before running
    - plot

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CTRNN simulation."
    )
    parser.add_argument(
        "-n", "--num_points",
        metavar="int",
        type=int,
        default=100,
        help="N for the NxN grid",
    )
    parser.add_argument(
        "-dt", "--timestep",
        metavar="float",
        default=1.0,
        help="Size of timesteps"
    )
    parser.add_argument(
        "-T", "--num_eval",
        metavar="int",
        default=100,
        help="Number of evaluation timesteps"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="filename",
        required=True,
        type=argparse.FileType("wb"),
        help="The filename where the simulation results should be written",
    )
    args = parser.parse_args()

    execute_simulation(args)

    # sim_func = partial(
    #     run_function,
    #     parsed args here
    # )

    # cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    #print("Launching simulation with following parameters:")
    #print(f"Number of CPU cores: {cpu_count}")
    #pprint(vars(args))
    # with mp.Pool(cpu_count) as p:
    #     training_outputs = p.map(training_func, range(n_random_seeds), chunksize=1)

    # save_training_stats(training_outputs, args.output)
