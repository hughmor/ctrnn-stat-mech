#!/home/plasticity/.virtualenvs/default/bin/python

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
import os
import logging
import tqdm
import argparse
from copy import deepcopy
from tqdm import tqdm
import json
import time
import datetime
import logging
import util
import simulation
import noise as noise_module

def run_temperature_sweep(temperatures, params, filenames):
    num_temps = len(temperatures)
    print(f'running {num_temps} simulations, the following is for *each* simulation:')
    util.check_disk_space_prompt_user(params)
    print(f'running...')


    # set up logging to print debug messages to the console
    #logging.basicConfig(level=logging.DEBUG)

    # Run the simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        #f = execute_with_params # dill.loads(dill.dumps(execute_with_params))

        total_items = num_temps  * params['num_eval']
        params_iter = (dict(temperature=temperature, output=filename, **params) for i, (temperature, filename) in enumerate(zip(temperatures, filenames)))
        #futures = [executor.submit(execute_simulation, **p) for p in params_iter]
        process_map(execute_simulation_with_params, params_iter)


        # wait for all the simulations to finish
        # print(f'waiting for simulations to finish...')
        # for future in concurrent.futures.as_completed(futures):
        #     try:
        #         result = future.result()
        #         print(f'result: {result}')
        #     except Exception as e:
        #         print(f'error getting pids and ports: {str(e)}')
        #         logging.error(f"Error processing item: {e}")
        #         import traceback
        #         traceback.print_exc()

        # print('getting pids and ports')
        # try:
        #     pids_ports = [(f._process.pid, f._process._popen.args[-1].split(':')[-1]) for f in futures]
        #     print('received pids and ports')
        # except Exception as e:
        #     print(f'error getting pids and ports: {str(e)}')
    


    # for filename, temperature in zip(filenames, temperature_sweep):
    #     params['temperature'] = temperature
    #     params['output'] = None
    #     execute_simulation(**params)

def execute_simulation_with_params(params):
    return execute_simulation(**params)

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
        coupling_matrix=None,
        init=None,
        verbose=True,
        log=False,
        overwrite_old_file=False,
        seed=None,
        inhibitory=False,
        print_tqdm=True,
        input_signals=None,
        ):

    if log:
        print_out = logging.info
    elif verbose:
        print_out = print
    else:
        print_out = lambda x: None
    
    # check if output filename is valid
    output = util.check_filename(output, overwrite=overwrite_old_file)
    # Tee(output+'.log', 'w') # good for logging but messes with displaying input()

    if parallel:
        if num_procs is None:
            num_procs = os.cpu_count()
    else:
        num_procs = 1

    # save metadata
    metadata = locals()
    
    # prompt user to proceed with simulation
    if not skip_prompt: util.check_disk_space_prompt_user(metadata)

    print_out("setting up grid and simulation...")
    # set numpy random seed
    if seed is not None:
        np.random.seed(seed)
        # this is apparently best practice for numpy random seed
        # from numpy.random import MT19937, RandomState, SeedSequence
        # rs = RandomState(MT19937(SeedSequence(123456789)))

    N = M = num_points
    biases = simulation.init_biases(N, M, dtype, distribution=init) # set up biases according to some random distribution
    grid = simulation.create_grid(biases) # initialize grid to biases
    
    # set up time parameters
    t_span = (0.0, num_eval * timestep)
    t_eval = np.linspace(t_span[0], t_span[1], num_eval + 1)

    # set up coupling matrix
    if coupling_matrix is None:
        coupling = simulation.create_coupling_matrix(coupling_radius, dtype, norm=coupling_sum, inhibitory=inhibitory)
        # gaussian rather than NN because I want to investigate the effect of coupling strength and radius
    else:
        coupling = coupling_matrix

    # run simulation over a number of timesteps
    #noise = (np.random.normal(0, temperature, biases.shape) for _ in range(num_eval)) # does noise need to have its variance scaled by sqrt(timestep)?
    noise_obj = noise_module.OrnsteinUhlenbeckActionNoise(mu=np.zeros_like(biases), sigma=temperature, dt=timestep)
    noise = (noise_obj() for _ in range(num_eval))

    results = []
    
    # set up external input
    ext_in = np.zeros_like(biases)
    if input_signals is None: input_signals = {}

    # start timer
    start = time.time()
    start_str = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H-%M-%S')
    metadata['start_time'] = start_str
    
    num_proc_str = f"{num_procs} processes" if num_procs != 1 else "1 process"
    sim_type = "parallel" if parallel else "serial"
    
    print_out(f"running {sim_type} simulation using {num_proc_str}...")
    results.append(deepcopy(grid))

    if print_tqdm:
        time_iter = tqdm(t_eval[:-1], desc=f"{os.getpid()}")
    else:
        time_iter = t_eval[:-1]

    for t in time_iter: # use tqdm to show progress bar
        for coord, signal in input_signals.items():
            x, y = coord
            ext_in[x, y] = signal(t)

        params = (coupling, next(noise), ext_in, biases)
        if parallel:
            res = simulation.parallel_simulation(grid, params, (t, t+timestep), num_chunks=num_procs)
        else:
            res = simulation.serial_simulation(grid, params, (t, t+timestep))

        results.append(deepcopy(res))
        grid = res
    
    # end timer
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d_%H-%M-%S')
    metadata['end_time'] = end_str

    print_out(f"finished! total time: {end - start:.2f} seconds")
    # save the results
    if output is not None:
        print_out(f"saving results to {output} ...")
        np.save(output, results)

        # removing input signals to be json serializable
        del metadata["input_signals"]

        with open(f"{output}.metadata", "w") as f_metadata:
            json.dump(metadata, f_metadata, default=repr)
        print_out("done!")
    else:
        return results, t_eval

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
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print information",
    )
    parser.add_argument(
        "-s", "--seed",
        metavar="int",
        type=int,
        default=None,
        help="Random seed",
    )

    args = parser.parse_args()
    args = vars(args)

    execute_simulation(**args)
