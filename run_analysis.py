import numpy as np
import datetime
from plotting import animate_grid, plot_grid
import util

def temperaure_sweep_data(date): # could maybe get this from metadata
    num_temps = 16
    temperature_sweep = np.logspace(-1, 1, num_temps)
    filenames = [f'temperature_sweep_{i}'for i in range(len(temperature_sweep))]
    results = [np.load(f'data/{date}/{filename}.npy') for filename in filenames]
    times = [util.get_t_eval_from_filename(f'data/{date}/{filename}.npy') for filename in filenames]

    return temperature_sweep, filenames, results, times

def animate_temperature_sweep(date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results, times = temperaure_sweep_data(date)

    # animate each of the results and save
    for i, (temperature, filename, result, t_eval) in enumerate(zip(temperature_sweep, filenames, results, times)):
        animate_grid(result, time=t_eval, filename=f'media/{date}/{filename}.mp4', title=f'Temperature: {temperature:.2f}')

def plot_temperature_sweep(date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results, times = temperaure_sweep_data(date)

    # plot the equilibrium state for each of the results and save
    for i, (temperature, filename, result) in enumerate(zip(temperature_sweep, filenames, results)):
        plot_grid(result, filename=f'media/{date}/{filename}', time_idx=-1, title=f'Temperature: {temperature:.2f}')

def animate_single_temperature(idx=0, date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results, times = temperaure_sweep_data(date)
    temp = temperature_sweep[idx]
    filename = filenames[idx]
    t_eval = times[idx]

    # animate the results
    animate_grid(results, time=t_eval, filename=f'media/{date}/{filename}.mp4', title=f'Temperature: {temp:.2f}')

def plot_single_temperature(idx=0, date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results, times = temperaure_sweep_data(date)
    temp = temperature_sweep[idx]
    filename = filenames[idx]

    # plot the equilibrium state
    plot_grid(results, filename=f'media/{date}/{filename}', time_idx=-1, title=f'Temperature: {temp:.2f}')

if __name__ == '__main__':
    animate_temperature_sweep()