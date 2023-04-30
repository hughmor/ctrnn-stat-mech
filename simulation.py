import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor
import pickle

from context import sim_context

def activity(results):
    # results should be the 2D grid at a single time
    return np.mean(np.tanh(results))

def energy(states, biases, only_multi=False):
    # states should be the 2D grid at a single time
    spins = np.tanh(states)
   
    coupling = sim_context['coupling_matrix']
    mid = sim_context['coupling_radius'] // 2
    self_weight = coupling[mid, mid]

    #energy = spins * (states - biases) + 0.5 * np.log(1 - spins * spins) -apply_weights(spins, coupling, parallel=False) + 0.5 * self_weight * spins * spins
    alt_energy = states*states/2 - states*biases - states * apply_weights(spins, coupling, parallel=False)

    return np.sum(alt_energy)/(sim_context['num_points']**2) # energy per lattice site

### Simulation state and saving/loading
class SimulationState():
    def __init__(self, metadata=None):
        self.state = {
            'time': [],
            'activity': [],
            'energy': [],
        }
        if sim_context['save_grid']:
            self.state['grid'] = []

        self.metadata = metadata

    def __getitem__(self, key):
        return self.state[key]
    
    def record(self, time, grid, biases):
        self.state['time'].append(time)
        if sim_context['save_grid']:
            self.state['grid'].append(grid)
        self.state['activity'].append(activity(grid))
        self.state['energy'].append(energy(grid, biases))

    def add_metadata(self, metadata):
        self.metadata = metadata

    def save(self, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'rb') as f:
            return pickle.load(f)

def save_results(filename, simulation_state):
    simulation_state.save(filename)

def load_results(filename):
    simulation_state = SimulationState.load(filename)
    return simulation_state

### Parallel simulation

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

def parallel_simulation(initial_conditions, params, t_span, num_chunks):
    kernel_size = params[0].shape[0]
    padding = kernel_size // 2
    grid_chunks = divide_grid(initial_conditions, num_chunks, padding=padding) # this one needs to know the kernel size so it can be padded
    
    noise = params[1]
    noise_chunks = divide_grid(noise, num_chunks) # this one doesn't need to know the kernel size

    ext_in = params[2]
    in_chunks = divide_grid(ext_in, num_chunks) # this one doesn't need to know the kernel size

    biases = params[3]
    bias_chunks = divide_grid(biases, num_chunks) # this one doesn't need to know the kernel size

    # Create a list of parameters for each process
    # add the bias chunk to the params
    proc_params = ((params[0], noise_chunk, in_chunk, bias_chunk,) for noise_chunk, in_chunk, bias_chunk in zip(noise_chunks, in_chunks, bias_chunks))
    timespans = (t_span for _ in range(num_chunks))
    parallel_arg = (True for _ in range(num_chunks))

    # Run the simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        ret = executor.map(run_simulation_step, grid_chunks, proc_params, timespans, parallel_arg)
        results = list(ret)

    # Recombine the chunks
    grid_state = recombine_chunks(results, biases.shape, num_chunks)

    return grid_state

### Serial simulation

def serial_simulation(initial_conditions, params, t_span):
    # Run simulation without parallelization
    sol = run_simulation_step(initial_conditions, params, t_span, parallel=False)
    return sol

### Initialization functions

def gaussian_kernel(N, dtype=None, norm=1.0):
    if dtype is None:
        dtype = sim_context.dtype
    dtype = np.dtype(dtype)
    
    # create an NxN matrix with gaussian distribution from the center
    matrix = np.zeros((N,N), dtype=dtype)
    for i in range(N):
        for j in range(N):
            matrix[i,j] = np.exp(-((i-N//2)**2 + (j-N//2)**2)/2)
    # normalize the matrix to sum to norm
    if norm is not None: # pass None to leave matrix unnormalized
        matrix = matrix / np.sum(matrix) * norm 
    return matrix

def create_grid(biases):
    # initial grid is just a copy of the biases
    return biases.copy()

def init_biases(nx, ny, dtype=None, init_bias=None): 
    if dtype is None:
        dtype = sim_context.dtype
    dtype = np.dtype(dtype)

    # Define the grid based on a chosen distribution
    # Default to uniform distribution on [-1, 1]
    if init_bias is not None:
        biases_array = init_bias.astype(dtype)
        assert biases_array.shape == (nx, ny), f"init_bias shape {biases_array.shape} does not match biases shape {(nx, ny)}"
    else:
        distribution = sim_context.bias_distribution
        biases_array = np.zeros((nx, ny), dtype=dtype)
        if distribution is None:
            distribution = sim_context.bias_distribution
        if distribution == 'uniform':
            biases_array[:] = 2*np.random.rand(nx, ny) - 1
        elif distribution == 'gaussian' or distribution == 'normal':
            biases_array[:] = np.random.normal(0, 1, (nx, ny))
        elif distribution == 'zero':
            pass
        else:
            raise ValueError(f"invalid bias distribution: {distribution}")
    return biases_array

### Main stepping functions

def apply_weights(s, coupling, parallel=True):
    # s is the state of the grid (NxN), coupling is the coupling matrix (MxM) M < N
    bcs = sim_context.boundary_conditions
    
    if bcs == 'periodic':
        if not parallel:
            # if serial simulation, then s is the whole grid
            # wrap boundary takes into account PBCs
            return convolve2d(s, coupling, mode='same', boundary='wrap')
        else:
            # if parallel simulation, then s is a chunk of the grid padded with adjacent cells
            # in this case, we want the convolution to return a smaller array and not wrap around
            return convolve2d(s, coupling, mode='valid', boundary='fill', fillvalue=0)
    elif bcs == 'free':
        if not parallel:
            # don't use boundary='wrap' because it will wrap around the edges
            return convolve2d(s, coupling, mode='same', boundary='fill', fillvalue=0)
        else:
            # if parallel simulation, then s is a chunk of the grid padded with adjacent cells
            # in this case, we want the convolution to return a smaller array and not wrap around
            raise NotImplementedError("TODO: free boundary conditions should be fixed in chunking function")
            return convolve2d(s, coupling, mode='valid', boundary='fill', fillvalue=0)
    else:
        raise ValueError(f"invalid boundary condition: {bcs}")

def ds_dt(t, s, inp):
    return -s + inp

def run_simulation_step(chunk, params, t_span, parallel=True):
    # unpack parameters
    coupling, noise, ext_in, bias = params
    
    # external input to the grid comes from the bias, the noise, external excitiation and the coupling from adjacent cells
    inp = bias + noise + ext_in + apply_weights(np.tanh(chunk), coupling, parallel=parallel)

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

    # Reshape the solution to the original shape of the chunk
    # ret_1d[:,0] is the initial condition
    # ret_1d[:,-1] is the final condition
    ret = ret_1d[:,-1].reshape(chunk.shape)
    
    return ret

