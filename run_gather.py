#!/home/plasticity/.virtualenvs/default/bin/python

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
import time
import datetime
from util import check_disk_space_prompt_user, check_filename, Tee


def create_grid(biases):
    # initial grid is just a copy of the biases
    return biases.copy()

def init_biases(nx, ny, dtype, distribution=None):
    # get numpy dtype object from string
    dtype = np.dtype(dtype)

    # Define the grid based on a chosen distribution
    # Default to uniform distribution on [-1, 1]
    initial_conditions = np.zeros((nx, ny), dtype=dtype)
    if distribution is None or distribution == 'uniform':
        initial_conditions[:] = 2*np.random.rand(nx, ny) - 1
    elif distribution == 'gaussian' or distribution == 'normal':
        initial_conditions[:] = np.random.normal(0, 1, (nx, ny))
    elif distribution == 'zero':
        pass
    else:
        raise ValueError(f"invalid bias distribution: {distribution}")
    return initial_conditions

def do_weighting(s, coupling, parallel=True):
    # s is the state of the grid (NxN), coupling is the coupling matrix (MxM) M < N
    if not parallel:
        # if serial simulation, then s is the whole grid
        # wrap boundary takes into account PBCs
        # TODO: for not PBC
        return convolve2d(s, coupling, mode='same', boundary='wrap')
    else:
        # if parallel simulation, then s is a chunk of the grid padded with adjacent cells
        # in this case, we want the convolution to return a smaller array and not wrap around
        return convolve2d(s, coupling, mode='valid', boundary='fill', fillvalue=0)

def ds_dt(t, s, inp):
    return -s + inp

def run_simulation(chunk, params, t_span, parallel=True):
    # unpack parameters
    coupling, temperature, bias = params

    # noise is an array of the same shape as bias, drawn from a normal distribution
    noise = np.random.normal(0, temperature, bias.shape)
    
    # external input to the grid comes from the bias, the noise, and the coupling from adjacent cells
    inp = bias + noise + do_weighting(np.tanh(chunk), coupling, parallel=parallel)

    if parallel:
        # discard the padding from the chunk
        kernel_size = params[0].shape[0]
        padding = kernel_size // 2
        chunk = chunk[padding:-padding, padding:-padding]

    # Flatten the chunk and input to a 1D array
    oned_chunk = chunk.ravel()
    oned_inp = inp.ravel()

    # Run the simulation for the given chunk
    sol = solve_ivp(ds_dt, t_span, oned_chunk, args=(oned_inp,), method='RK45')
    ret_1d = sol.y

    # check how many time steps were taken # TODO remove this
    num_time_steps = ret_1d.shape[1]
    if num_time_steps > 5:
        print(f"WARNING: took {num_time_steps} time steps at time {t_span[0]}")

    # Reshape the solution to the original shape of the chunk
    # ret_1d[:,0] is the initial condition
    # ret_1d[:,-1] is the final condition
    # TODO: right now, not saving the intermediate time steps
    ret = ret_1d[:,-1].reshape(chunk.shape)
    return ret

def serial_simulation(initial_conditions, params, t_span):
    # Run simulation without parallelization
    sol = run_simulation(initial_conditions, params, t_span, parallel=False)
    return sol

def parallel_simulation(initial_conditions, params, t_span, num_chunks):
    kernel_size = params[0].shape[0]
    padding = kernel_size // 2
    grid_chunks = divide_grid(initial_conditions, num_chunks, padding=padding) # this one needs to know the kernel size so it can be padded
    
    biases = params[2]
    bias_chunks = divide_grid(biases, num_chunks) # this one doesn't need to know the kernel size

    # Create a list of parameters for each process
    # add the bias chunk to the params
    proc_params = ((params[0], params[1], bias_chunk,) for bias_chunk in bias_chunks)
    timespans = (t_span for _ in range(num_chunks))
    parallel_arg = (True for _ in range(num_chunks))

    # Run the simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        ret = executor.map(run_simulation, grid_chunks, proc_params, timespans, parallel_arg)
        results = list(ret)

    # Recombine the chunks
    grid_state = recombine_chunks(results, biases.shape, num_chunks)

    return grid_state

def chunk(N):
    x, y = 1, N
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            other_factor = N // i
            if abs(i - other_factor) < abs(x - y):
                x, y = i, other_factor
    return x, y

def periodic_subarray(arr, x_start, x_end, y_start, y_end):
    x_size, y_size = arr.shape

    # Adjust the indices to be within the bounds of the array using modulo
    x_start_mod = x_start % x_size
    x_end_mod = x_end % x_size
    y_start_mod = y_start % y_size
    y_end_mod = y_end % y_size

    # Create the subarray with periodic boundary conditions
    if x_start_mod <= x_end_mod and y_start_mod <= y_end_mod:
        subarr = arr[x_start_mod:x_end_mod, y_start_mod:y_end_mod]
    elif x_start_mod > x_end_mod and y_start_mod <= y_end_mod:
        subarr = np.concatenate((arr[x_start_mod:, y_start_mod:y_end_mod], arr[:x_end_mod, y_start_mod:y_end_mod]), axis=0)
    elif x_start_mod <= x_end_mod and y_start_mod > y_end_mod:
        subarr = np.concatenate((arr[x_start_mod:x_end_mod, y_start_mod:], arr[x_start_mod:x_end_mod, :y_end_mod]), axis=1)
    else:
        subarr_x = np.concatenate((arr[x_start_mod:, :], arr[:x_end_mod, :]), axis=0)
        subarr = np.concatenate((subarr_x[:, y_start_mod:], subarr_x[:, :y_end_mod]), axis=1)

    return subarr

def divide_grid(grid, num_chunks, padding=0):
    # Divide the 2D grid into num_chunks total chunks
    cx, cy = chunk(num_chunks)
    nx, ny = grid.shape
    
    # Figure out the indices of each chunk
    # Take extra padding into account and use periodic boundary conditions to wrap around
    grid_chunks = []
    for i in range(cx):
        for j in range(cy):
            x_start = i * nx // cx - padding
            x_end = (i + 1) * nx // cx + padding
            y_start = j * ny // cy - padding
            y_end = (j + 1) * ny // cy + padding

            grid_chunk = periodic_subarray(grid, x_start, x_end, y_start, y_end)
            grid_chunks.append(grid_chunk)
    return grid_chunks

def recombine_chunks(chunks, shape, num_chunks):
    # Given a list of chunks, recombine them into a single array
    cx, cy = chunk(num_chunks)

    # Compute the total array size from the size of each chunk
    nx, ny = shape
    
    # Create an empty array to store the recombined chunks
    grid = np.zeros((nx, ny))

    # Loop over the chunks and recombine them
    for i in range(cx):
        for j in range(cy):
            x_start = i * nx // cx
            x_end = (i + 1) * nx // cx
            y_start = j * ny // cy
            y_end = (j + 1) * ny // cy

            grid[x_start:x_end, y_start:y_end] = chunks[i * cx + j]
    return grid

def create_coupling_matrix(N, dtype, norm=1.0):
    # create an NxN matrix with gaussian distribution from the center
    matrix = np.zeros((N,N), dtype=dtype)
    for i in range(N):
        for j in range(N):
            matrix[i,j] = np.exp(-((i-N//2)**2 + (j-N//2)**2)/2)
    # normalize the matrix to sum to norm
    if norm is not None: # pass None to leave matrix unnormalized
        matrix = matrix / np.sum(matrix) * norm 
    return matrix

def execute_simulation(
        output=None,
        num_points=100,
        num_eval=100,
        timestep=1.0,
        temperature=1.0,
        parallel=False,
        num_procs=None,
        skip_prompt=False,
        dtype='float32',
        coupling_radius=3,
        coupling_sum=1.0,
        init=None,
        **kwargs
        ):

    # check if output filename is valid
    output = check_filename(output)
    # Tee(output+'.log', 'w') # good for logging but messes with displaying input()
    
    if parallel:
        if num_procs is None:
            num_procs = os.cpu_count()
    else:
        num_procs = 1

    # save metadata
    metadata = locals()
    
    print("setting up grid and simulation...")
    N = M = num_points
    biases = init_biases(N, M, dtype, distribution=init) # set up biases according to some random distribution
    grid = create_grid(biases) # initialize grid to biases
    
    # set up time parameters
    t_span = (0.0, num_eval * timestep)
    t_eval = np.linspace(t_span[0], t_span[1], num_eval)

    # set up coupling matrix
    coupling = create_coupling_matrix(coupling_radius, dtype, norm=coupling_sum)
    # gaussian rather than NN because I want to investigate the effect of coupling strength and radius

    # prompt user to proceed with simulation
    if not skip_prompt: check_disk_space_prompt_user(grid, t_eval)

    # run simulation over a number of timesteps
    params = (coupling, temperature, biases)
    results = []
    
    # start timer
    start = time.time()
    start_str = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H-%M-%S')
    metadata['start_time'] = start_str
    
    num_proc_str = f"{num_procs} processes" if num_procs != 1 else "1 process"
    sim_type = "parallel" if parallel else "serial"
    print(f"running {sim_type} simulation using {num_proc_str}...")

    for t in tqdm(t_eval, desc="     "): # use tqdm to show progress bar
        if parallel:
            res = parallel_simulation(grid, params, (t, t+1), num_chunks=num_procs)
        else:
            res = serial_simulation(grid, params, (t, t+1))
        results.append(deepcopy(res))
        grid = res
    
    # end timer
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d_%H-%M-%S')
    metadata['end_time'] = end_str

    print(f"finished! total time: {end - start:.2f} seconds")
    # save the results
    if output is not None:
        print(f"saving results to {output} ...")
        np.save(output, results)
        with open(f"{output}.metadata", "w") as f_metadata:
            json.dump(metadata, f_metadata, default=repr)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTRNN simulation.")
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
        type=float,
        default=1.0,
        help="Size of timesteps"
    )
    parser.add_argument(
        "-t", "--num_eval",
        metavar="int",
        type=int,
        default=100,
        help="Number of evaluation timesteps"
    )
    parser.add_argument(
        "-T", "--temperature",
        metavar="float",
        type=float,
        default=1.0,
        help="Temperature of the simulation (standard deviation of the noise process)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="filename",
        required=True,
        help="The filename where the simulation results should be written",
    )
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run the simulation in parallel",
    )
    parser.add_argument(
        "-y", "--skip_prompt",
        action="store_true",
        help="Skip the prompt to check disk space",
    )
    parser.add_argument(
        "-np", "--num_procs",
        metavar="int",
        type=int,
        default=None,
        help="Number of processes to use for parallel simulation",
    )
    parser.add_argument(
        "-d", "--dtype",
        metavar="str",
        type=str,
        default="float64",
        help="Numpy datatype to use for simulation",
    )
    parser.add_argument(
        "--coupling_radius",
        metavar="int",
        type=int,
        default=3,
        help="Radius of the coupling matrix",
    )
    parser.add_argument(
        "--coupling_sum",
        metavar="float",
        type=float,
        default=1.0,
        help="Sum of the coupling matrix values",
    )
    parser.add_argument(
        "--init",
        metavar="str",
        type=str,
        default="uniform",
        help="Initial bias coniditons (uniform, normal, or zero)",
    )

    args = parser.parse_args()
    args = vars(args)

    execute_simulation(**args)

    # cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    #print("Launching simulation with following parameters:")
    #print(f"Number of CPU cores: {cpu_count}")
    #pprint(vars(args))
    # with mp.Pool(cpu_count) as p:
    #     training_outputs = p.map(training_func, range(n_random_seeds), chunksize=1)

    # save_training_stats(training_outputs, args.output)
