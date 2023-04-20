import sys
import numpy as np
import datetime
import os

def format_bytes(bits):
    # this function should take an integer number of bits and return a string with a human readable string for the size with the best prefix
    prefixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    prefix_index = int(np.floor(np.log2(bits)/10))
    prefix = prefixes[prefix_index]
    size = bits / 2**(10*prefix_index)
    size_fmtd = f"{size:.2f}"

    # return the size with the prefix
    return f"{size_fmtd} {prefix}"

def check_disk_space(grid, time):
    # get number of grid points and their data types
    # multiply num_points x size/point x num_timesteps
    dtype = grid.dtype
    num_grid = grid.shape[0] * grid.shape[1]
    num_t = len(time)
    
    type_to_bits = {
        np.dtype('float64'): 64,
        np.dtype('float32'): 32,
        np.dtype('float16'): 16,
    }

    size_bits = num_t * num_grid * type_to_bits[dtype]
    size_bytes_fmtd = format_bytes(size_bits/8)

    return size_bytes_fmtd

def check_disk_space_prompt_user(grid, time):
    disk_space = check_disk_space(grid, time)

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
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    if output is None:
        if input("no output file specified. are you sure you want to continue wihtout saving? (y/n) ") != "y":
            sys.exit("no output file specified. exiting...")
        else:
            return output
    elif os.path.isabs(output): # if output is a fully qualified path, use that
        output = output
    elif os.path.isdir("data"): # otherwise, only proceed if the data directory exists
        if not os.path.isdir(os.path.join("data", date)): # make a subdirectory based on the current date
            os.mkdir(os.path.join("data", date))
        output = os.path.join("data", date, output)
        if not output.endswith('.npy'): output += '.npy'
    else:
        sys.exit("no data directory found. exiting...")
    if os.path.isfile(output):
        resp = input(f"file {output} already exists. overwrite? (Y/n) ")
        if resp.lower() == "n":
            sys.exit("exiting...")
    return output

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