import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor

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

def create_coupling_matrix(N, dtype, norm=1.0, inhibitory=False):
    # create an NxN matrix with gaussian distribution from the center
    matrix = np.zeros((N,N), dtype=dtype)
    for i in range(N):
        for j in range(N):
            matrix[i,j] = np.exp(-((i-N//2)**2 + (j-N//2)**2)/2)
    if inhibitory:
        matrix[N // 2, N // 2] *= -1
    # normalize the matrix to sum to norm
    if norm is not None: # pass None to leave matrix unnormalized
        matrix = matrix / np.sum(matrix) * norm 
    return matrix

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
