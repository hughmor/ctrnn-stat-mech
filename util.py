import sys
import numpy as np
import datetime
import os
import json

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

def check_disk_space_prompt_user(metadata):
    num_points = metadata['num_points']
    num_eval = metadata['num_eval']
    dtype = metadata['dtype']

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

def check_filename(output, overwrite=False):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
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
        if not output.endswith('.npy'): output += '.npy'
        # also make a subdirectory for the current date in the media folder
        if not os.path.isdir(os.path.join("media", date)):
            os.mkdir(os.path.join("media", date))
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

def get_t_eval_from_filename(filename):
    metadata = parse_metadata(filename)
    num_eval = metadata['num_eval']
    timestep = metadata['timestep']
    t_span = (0.0, num_eval * timestep)
    return np.linspace(t_span[0], t_span[1], num_eval + 1)

class Tee(object):
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