#%%
import numpy as np
import matplotlib.pyplot as plt
from run_gather import create_coupling_matrix


#%%
def test_coupling_matrices():
    # generate a few coupling matrices and plot them using imshow
    fig, ax = plt.subplots(1,3)
    for i in range(3):
        coupling = create_coupling_matrix(2*(i+1)+1)
        ax[i].imshow(coupling, cmap='gray')

    plt.show()

#%%

size = range(3,12,2)
for i in size:
    print(i)
    coupling = create_coupling_matrix(i)
    print(coupling.sum())
# %%

from noise import OrnsteinUhlenbeckActionNoise

# create a set of noise processes
sigmas = [0.02, 0.2, 2.0]
thetas = [0.02, 0.15, 1.2]
dts = [1e-1, 1e-2, 1e-3]

# create noise processes with all combinations of the above parameters
noise_processes = {}
for sigma in sigmas:
    for theta in thetas:
        for dt in dts:
            noise_processes[(sigma, theta, dt)] = OrnsteinUhlenbeckActionNoise(sigma=sigma, theta=theta, dt=dt)

# evaluate the noise processes over 1 second each
t = np.linspace(0, 1, 1000)
noise_results = {}
for key, noise in noise_processes.items():
    noise_results[key] = [noise() for _ in range(len(t))]

# plot the noise processes
fig, ax = plt.subplots(3,3, dpi=200, figsize=(10,10))
for i, sigma in enumerate(sigmas):
    for j, theta in enumerate(thetas):
        for k, dt in enumerate(dts):
            ax[i,j].plot(t, noise_results[(sigma, theta, dt)], label=f'dt={dt}')
        ax[i,j].set_title(f'sigma={sigma}, theta={theta}')
        ax[i,j].legend()

# %%

# plot the fourier transform of the noise processes
fig, ax = plt.subplots(3,3, dpi=200, figsize=(10,10))
for i, sigma in enumerate(sigmas):
    for j, theta in enumerate(thetas):
        for k, dt in enumerate(dts):
            fft = np.fft.fft(noise_results[(sigma, theta, dt)])
            N = len(fft)
            n = np.arange(N)
            T = N/1000
            freq = n/T
            ax[i,j].plot(freq, np.abs(fft), label=f'dt={dt}')
        ax[i,j].set_title(f'sigma={sigma}, theta={theta}')
        ax[i,j].legend()
# %%


