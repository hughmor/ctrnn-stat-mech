import sys
import numpy as np
import datetime
import os
import json
import logging
import time
from tqdm import tqdm

from context import sim_context


def start_timer():
    start = time.time()
    start_str = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H-%M-%S')
    sim_context.start_time = start_str
    return start

def end_timer(start):
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d_%H-%M-%S')
    sim_context.end_time = end_str
    sim_context.elapsed_time = end - start
    return end

def get_time_iterable(t):
    if sim_context.tqdm:
        time_iterator = tqdm(t[:-1], desc=f"{os.getpid()}")
    else:
        time_iterator = t[:-1]
    return time_iterator

def format_bytes(bits):
    # this function should take an integer number of bits and return a string with a human readable string for the size with the best prefix
    prefixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    prefix_index = int(np.floor(np.log2(bits)/10))
    prefix = prefixes[prefix_index]
    size = bits / 2**(10*prefix_index)
    size_fmtd = f"{size:.2f}"

    # return the size with the prefix
    return f"{size_fmtd} {prefix}"

def check_disk_space(num_points, num_eval, dtype):
    # get number of grid points and their data types
    # multiply num_points x size/point x num_timesteps
    num_grid = num_points**2
    
    type_to_bits = {
        'float64': 64,
        'float32': 32,
        'float16': 16,
    }

    size_bits = (num_eval + 1) * num_grid * type_to_bits[dtype]
    size_bytes_fmtd = format_bytes(size_bits/8)

    return size_bytes_fmtd

def check_disk_space_prompt_user():
    num_points = sim_context['num_points']
    num_eval = sim_context['num_eval']
    dtype = sim_context['dtype']

    disk_space = check_disk_space(num_points, num_eval, dtype)

    user_response = input(f"the results file will take ~{disk_space} on disk. proceed? [Y/n]")
    valid_responses = {
        "y": lambda: None, # do nothing
        "n": lambda: sys.exit(), # exit the program
        "": lambda: None, # do nothing
    }
    while user_response.lower() not in valid_responses:
        user_response = input(f"invalid input. proceed? [Y/n]")
    valid_responses[user_response.lower()]()

def check_filename(output):
    overwrite = sim_context.overwrite_file
    date = sim_context.date
    if output is None:
        return output
    elif os.path.isabs(output): # if output is a fully qualified path, use that
        output = output
    elif os.path.isdir("data"): # otherwise, only proceed if the data directory exists
        if not os.path.isdir(os.path.join("data", date)): # make a subdirectory based on the current date
            os.mkdir(os.path.join("data", date))
        if os.path.dirname(output) != "": # check if the output file specifies a subdirectory and make it if it doesn't exist
            if not os.path.isdir(os.path.join("data", date, os.path.dirname(output))):
                os.mkdir(os.path.join("data", date, os.path.dirname(output)))
        output = os.path.join("data", date, output)
        #if not output.endswith('.npy'): output += '.npy'
    else:
        sys.exit("no data directory found. exiting...")
    if os.path.isfile(output) and not overwrite:
        resp = input(f"file {output} already exists. overwrite? (Y/n) ")
        if resp.lower() == "n":
            sys.exit("exiting...")
    return output

def parse_metadata(filename):
    # filename should end with .npy.metadata
    if filename.endswith('.metadata'):
        pass
    elif filename.endswith('.npy'):
        filename += '.metadata'
    else:
        filename += '.npy.metadata'

    # file was written with json.dump
    with open(filename, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_num_procs():
    parallel = sim_context.parallel
    num_procs = sim_context.num_procs
    if parallel:
        if num_procs is None:
            num_procs = os.cpu_count()
    else:
        num_procs = 1
    return num_procs

def write():
    log = sim_context.log
    verbose = sim_context.verbose
    if log:
        print_out = logging.info
    elif verbose:
        print_out = print
    else:
        print_out = lambda x: None
    return print_out

class Tee(object):
    # TODO: I liked how this class works but doesn't play nice with input()
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


def nonlinear_spacing(v1, v2, N, steepness=1, type='sigmoid'):
    if type == 'sin':
        linear_points = np.linspace(0, 1, N) # linear spacing between 0 and 1
        nonlinear_points = np.arcsin(2*linear_points - 1) / np.pi + 0.5 # nonlinear spacing between 0 and 1
        return v1 + (v2 - v1) * nonlinear_points # scale to v1 and v2
    elif type == 'sigmoid':
        midpoint = (v1 + v2)/2
        min_x = 1/(np.exp((midpoint-v1)*steepness) + 1)
        max_x = 1/(np.exp((midpoint-v2)*steepness) + 1)
        linear_points = np.linspace(min_x, max_x, N) # linear spacing between 0 and 1

        nonlinear_points = -np.log(1/linear_points - 1)/steepness + midpoint # nonlinear spacing between 0 and 1
        # clip to v1 and v2 - should be unnecessary, but maybe rounding causes some issues
        nonlinear_points[nonlinear_points < v1] = v1
        nonlinear_points[nonlinear_points > v2] = v2
        return nonlinear_points
