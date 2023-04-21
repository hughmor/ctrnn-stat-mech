import numpy as np
import datetime
from plotting import animate_grid, plot_grid

def temperaure_sweep_data(date): # could maybe get this from metadata
    num_temps = 16
    temperature_sweep = np.logspace(-1, 1, num_temps)
    filenames = [f'temperature_sweep_{i}'for i in range(len(temperature_sweep))]
    results = [np.load(f'data/{date}/{filename}.npy') for filename in filenames]

    return temperature_sweep, filenames, results

def animate_temperature_sweep(date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results = temperaure_sweep_data(date)

    # animate each of the results and save
    for i, (temperature, filename, result) in enumerate(zip(temperature_sweep, filenames, results)):
        animate_grid(result, f'media/{date}/{filename}.mp4', title=f'Temperature: {temperature:.2f}')

def plot_temperature_sweep(date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results = temperaure_sweep_data(date)

    # plot the equilibrium state for each of the results and save
    for i, (temperature, filename, result) in enumerate(zip(temperature_sweep, filenames, results)):
        plot_grid(result, f'media/{date}/{filename}', time_idx=-1, title=f'Temperature: {temperature:.2f}')

def animate_single_temperature(idx=0, date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results = temperaure_sweep_data(date)
    temp = temperature_sweep[idx]
    filename = filenames[idx]

    # animate the results
    animate_grid(results, f'media/{date}/{filename}.mp4', title=f'Temperature: {temp:.2f}')

def plot_single_temperature(idx=0, date=None):
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    temperature_sweep, filenames, results = temperaure_sweep_data(date)
    temp = temperature_sweep[idx]
    filename = filenames[idx]

    # plot the equilibrium state
    plot_grid(results, f'media/{date}/{filename}', time_idx=-1, title=f'Temperature: {temp:.2f}')

def main():
    # load the results
    filename = '2023-04-20/delme'
    results = np.load(f'data/{filename}.npy')

    # animate the results
    animate_grid(results, f'media/{filename}.mp4')

if __name__ == '__main__':
    animate_temperature_sweep()