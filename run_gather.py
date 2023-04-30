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
import itertools

import util
import simulation
import noise
import analysis
DEFAULT_CONFIG = 'sim_config.json'
from context import sim_context

def run_generalized_sweep(sweep_params, params, config_json, base_filename=None):
    sim_context.load_config(config_json)
    sim_context.update_config(params)

    # sweep params is a dictionary of lists, where each list is a sweep over a parameter
    # e.g. sweep_params = {'noise': [0.1, 0.2, 0.3], 'bias': [0.1, 0.2, 0.3]}
    # create a list of dictionaries, where each dictionary is a set of parameters for a simulation
    
    for param_name, param_values in sweep_params.items():
        if not isinstance(param_values, list):
            sweep_params[param_name] = [param_values]
    if base_filename is None:
        base_filename = 'master/' + '_'.join(sweep_params.keys())
    sweep_params = [dict(zip(sweep_params, v)) for v in itertools.product(*sweep_params.values())]
    filenames = [f'{base_filename}_{i}' for i in range(len(sweep_params))]   

    # Run the simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        params_iter = (dict(output_file=filename, **sweep_param, **params) for i, (sweep_param, filename) in enumerate(zip(sweep_params, filenames)))
        #futures = [executor.submit(execute_simulation, **p) for p in params_iter]
        process_map(execute_simulation_with_params, params_iter)

def run_temperature_sweep(temperatures, params, filenames, config_json):
    sim_context.load_config(config_json)
    sim_context.update_config(params)

    num_temps = len(temperatures)
    print(f'running {num_temps} simulations, the following is for *each* simulation:')
    util.check_disk_space_prompt_user()
    print(f'running...')

    # set up logging to print debug messages to the console
    #logging.basicConfig(level=logging.DEBUG)

    # Run the simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        params_iter = (dict(output_file=filename, temperature=temperature, **params) for i, (temperature, filename) in enumerate(zip(temperatures, filenames)))
        #futures = [executor.submit(execute_simulation, **p) for p in params_iter]
        process_map(execute_simulation_with_params, params_iter)

def execute_simulation_with_params(params):
    return execute_simulation(**params)

def execute_simulation(output_file=None,
                       input_signals=None,
                       initial_condition=None,
                       init_bias=None,
                       coupling_matrix=None,
                       skip_prompt=False,
                       config_json=None,
                       **kwargs):
    # set up metadata context
    if config_json is not None: # filename was passed in, assume this is what we want to load
        sim_context.load_config(config_json)
    else: # no config was passed in
        if sim_context._config is None: # first check if the global context has been loaded already
            sim_context.load_config(DEFAULT_CONFIG)
    sim_context.update_config(kwargs) # update the context with any additional kwargs

    # set up logging to print debug messages to the console
    print = util.write()
    
    # check if output filename is valid
    output_file = util.check_filename(output_file)

    # set up parallel processing
    num_procs = util.get_num_procs()
    if sim_context.parallel:
        simulation_step = lambda ic, pars, t: simulation.parallel_simulation(ic, pars, t, num_procs=num_procs)
    else:
        simulation_step = simulation.serial_simulation
    
    # prompt user to proceed with simulation
    if not skip_prompt: util.check_disk_space_prompt_user()

    # main simulation set up
    print("setting up grid and simulation...")
    if sim_context.seed is not None:
        np.random.seed(sim_context.seed)

    # set up simulation parameters
    num_points = sim_context.num_points
    num_eval = sim_context.num_eval
    timestep = sim_context.timestep
    temperature = sim_context.temperature
    coupling_radius = sim_context.coupling_radius
    coupling_sum = sim_context.coupling_sum

    biases = simulation.init_biases(num_points, num_points, init_bias=init_bias) # set up biases according to some random distribution

    if initial_condition is None or initial_condition == 'biases':
        grid = simulation.create_grid(biases) # initialize grid to biases as initial condition
    else:
        if initial_condition == 'random':
            grid = 2*np.random.rand(num_points, num_points).astype(sim_context.dtype) - 1
        elif isinstance(initial_condition, np.ndarray):
            grid = initial_condition
        assert grid.shape == biases.shape, f"initial condition shape {grid.shape} does not match biases shape {biases.shape}"

    if coupling_matrix is None:
        coupling_matrix = simulation.gaussian_kernel(coupling_radius, norm=coupling_sum) # set up coupling matrix
    else:
        sim_context['coupling_sum'] = coupling_matrix.sum()
        sim_context['coupling_radius'] = coupling_matrix.shape[0] // 2
    sim_context['coupling_matrix'] = coupling_matrix
        
    t_span = (0.0, num_eval * timestep)
    t_eval = np.linspace(t_span[0], t_span[1], num_eval + 1)
    
    noise_proc = noise.noise_process(temperature)
    simulation_state = simulation.SimulationState()

    # set up external input
    external_in = np.zeros_like(biases)
    if input_signals is None: input_signals = {}

    num_proc_str = f"{num_procs} processes" if num_procs != 1 else "1 process"
    sim_type = "parallel" if sim_context.parallel else "serial"
    print(f"running {sim_type} simulation using {num_proc_str}...")

    # start timer
    start = util.start_timer()
    # run simulation step by step
    time_range = util.get_time_iterable(t_eval)
    for t in time_range:
        for (x, y), signal in input_signals.items():
            external_in[x, y] = signal(t)

        simulation_state.record(t, deepcopy(grid), biases)
        params = (coupling_matrix, next(noise_proc), external_in, biases)
        grid = simulation_step(grid, params, (t, t+timestep))        
        
        if sim_context.end_when_steady and t > t_span[1]/4: # at least 25% of the way through simulation
            if analysis.simulation_steady(simulation_state):
                break
            
    simulation_state.record(t+timestep, deepcopy(grid), biases) # add final state

    # end timer
    end = util.end_timer(start)

    # finish up
    print(f"finished! total time: {end - start:.2f} seconds")
    
    simulation_state.add_metadata(sim_context._config)
    if output_file is not None:
        print(f"saving results to {output_file} ...")
        simulation.save_results(output_file, simulation_state)
        print("done!")
    else:
        return simulation_state


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
        "-o", "--output-file",
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
        "--bias_distribution",
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
