import sys
import numpy as np


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
    }

    size_bits = num_t * num_grid * type_to_bits[dtype]
    size_bytes_fmtd = format_bytes(size_bits/8)

    return size_bytes_fmtd

def check_disk_space_prompt_user(grid, time):
    disk_space = check_disk_space(grid, time)

    user_response = input(f"the results file will take ~{disk_space} on disk. proceed? [Y/n]")
    valid_responses = {
        "y": lambda: None, # do nothing
        "n": lambda: sys.exit() # exit the program
    }
    while user_response.lower() not in valid_responses:
        user_response = input(f"invalid input. proceed? [Y/n]")
    valid_responses[user_response]()