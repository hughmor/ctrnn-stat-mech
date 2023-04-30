
#%%

import simulation
import util
import analysis
from run_gather import run_generalized_sweep, run_temperature_sweep, execute_simulation, execute_simulation_with_params
from run_analysis import animate_grid, plot_grid

import numpy as np

T = 50
N = 11

bias = np.random.normal(0.1, 0.1, (N,N))

weight_norm = 1.0
coupling_matrix = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)*weight_norm

pars = dict(
    num_points=N,
    num_steps=T,
    timestep=0.1,
    init_bias=bias,
    initial_condition='biases',
    boundary_conditions='periodic',
    skip_prompt=True,
    coupling_matrix=coupling_matrix,
    save_grid=True,
    temperature=0.1
)

# read in the seeds from a file
with open('seeds.txt', 'r') as f: # 1000 seeds in file
    seeds = [int(line.strip()) for line in f.readlines()][:10]

# sweep temperatures in the sweeps
# num_temps = 30
# temperatures = util.nonlinear_spacing(0, 15, num_temps, steepness=0.75).tolist()
num_temps = 30
temperatures = np.linspace(0, 15, num_temps).tolist()

sizes = [25, 50, 100]

sweep_results = {}
for size in sizes:
    pars['num_points'] = size
    bias = np.random.normal(0.1, 0.1, (size,size))
    pars['init_bias'] = bias
    sweep_results[size] = {}
    for temp in temperatures:
        pars['temperature'] = temp
        sweep_results[size][temp] = {}
        for seed in seeds:
            pars['seed'] = seed
            sweep_results[size][temp][seed] = execute_simulation(**pars)
        

#state = execute_simulation(**pars)
#%%

"""   HEAT CAPACITY ANALYSIS   """

from context import sim_context

for size in sizes:
    for temp in temperatures:
        final_es = []
        heat_caps = []
        for seed in seeds:
            result = sweep_results[size][temp][seed]
            #sim_context.init_from_dict(result.metadata)

            m = result.state['activity']
            e = result.state['energy']
            t = result.state['time']

            heat_cap_in_trial = analysis.heat_capacity(e[-T//2:])
            heat_caps.append(heat_cap_in_trial)
            final_es.append(np.mean(np.abs(e[-T//2:])))

        sim_context.init_from_dict(result.metadata)

        heat_cap_across_trials = analysis.heat_capacity(final_es)
        sweep_results[size][temp]['heat_cap'] = heat_cap_across_trials * (size**2)


%matplotlib inline
import matplotlib.pyplot as plt
plt.figure()
tc = 0.0
for size in sizes:
    plt.plot(temperatures, [sweep_results[size][temp]['heat_cap'] for temp in temperatures], '-', label=f'N={size}')
    maximum_index = np.argmax([sweep_results[size][temp]['heat_cap'] for temp in temperatures])
    tc += temperatures[maximum_index]
tc /= len(sizes)

plt.axvline(tc, color='k', linestyle='--')
plt.text(tc+0.5, 0.5, f'$T_c$ = {tc:.2f}', va='bottom', ha='left')

plt.xlabel('Temperature ($T$)')
plt.ylabel('Specific Heat ($C$)')
plt.legend()
plt.show()


#%%

"""   SUCCEPTIBILITY ANALYSIS   """

for size in sizes:
    for temp in temperatures:
        final_ms = []
        succs = []
        for seed in seeds:
            result = sweep_results[size][temp][seed]
            #sim_context.init_from_dict(result.metadata)

            m = result.state['activity']
            e = result.state['energy']
            t = result.state['time']

            succ_in_trial = analysis.succeptability(m[-T//2:])
            succs.append(succ_in_trial)
            final_ms.append(np.mean(np.abs(m[-T//2:])))

        sim_context.init_from_dict(result.metadata)

        succ_across_trials = analysis.succeptability(final_ms)
        sweep_results[size][temp]['succeptability'] = succ_across_trials * (size**2)

%matplotlib inline
import matplotlib.pyplot as plt
plt.figure()
tc = 0.0
for size in sizes:
    plt.plot(temperatures, [sweep_results[size][temp]['succeptability'] for temp in temperatures], '-', label=f'N={size}')
    maximum_index = np.argmax([sweep_results[size][temp]['succeptability'] for temp in temperatures])
    tc += temperatures[maximum_index]
tc /= len(sizes)

plt.axvline(tc, color='k', linestyle='--')
plt.text(tc+0.5, 0.5, f'$T_c$ = {tc:.2f}', va='bottom', ha='left')

plt.xlabel('Temperature ($T$)')
plt.ylabel('Succeptibility ($\chi$)')
plt.legend()
plt.show()


#%%

"""   DOMAINS ANALYSIS    """

for size in sizes[-1:]:
    for temp in temperatures:
        domain_numbers = []
        domain_sizes = []
        for seed in seeds:
            result = sweep_results[size][temp][seed]
            #sim_context.init_from_dict(result.metadata)

            domains = analysis.find_domains(result.state['grid'][-1]) # list of length=num_domains, each element is the size of that domain
            domain_numbers.append(len(domains)) # number of domains for this temperature and seed
            domain_sizes.append(np.mean(domains)) # mean domain size for this temperature and seed

        sweep_results[size][temp]['num_domains'] = np.mean(domain_numbers)
        sweep_results[size][temp]['mean_domain_size'] = np.mean(domain_sizes)
    
        domain_succ = np.var(domain_sizes) / temp
        sweep_results[size][temp]['domain_succeptability'] = domain_succ / (size**4)

#%%
%matplotlib inline
import matplotlib.pyplot as plt

plt.figure()
#for size in sizes[-1:]:
plt.plot(temperatures, [sweep_results[sizes[1]][temp]['num_domains'] for temp in temperatures], '-', label=f'N={size}')
plt.xlabel('Temperature ($T$)')
plt.ylabel('Number of Domains ($\mathcal{N}_D$)')
#plt.legend()
plt.show()

plt.figure()
#for size in sizes[-1:]:
plt.plot(temperatures, [sweep_results[sizes[1]][temp]['mean_domain_size'] for temp in temperatures], '-', label=f'N={size}')
plt.xlabel('Temperature ($T$)')
plt.ylabel('Mean Domain Size ($\mathcal{D}$)')
#plt.legend()
plt.show()

# plt.figure()
# for size in sizes[-1:]:
#     plt.plot(temperatures, [sweep_results[size][temp]['domain_succeptability'] for temp in temperatures], '-', label=f'N={size}')
# plt.xlabel('Temperature ($T$)')
# plt.ylabel('Domain Succeptability ($\chi$)')
# plt.legend()
# plt.show()


#%%
%matplotlib inline
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize=(8,8))
size = sizes[-1]
seed = seeds[0]
temp_indices = [0, 25, -1]

for i, temp_index in enumerate(temp_indices):
    temp = temperatures[temp_index]
    result_state = sweep_results[size][temperatures[temp_index]][seed].state['grid']


    axs[i, 0].imshow(np.tanh(result_state[0]), vmin=-1, vmax=1)
    axs[i, 0].axis('off')
    
    axs[i, 1].imshow(np.tanh(result_state[T//2]), vmin=-1, vmax=1)
    axs[i, 1].axis('off')
    
    axs[i, 2].imshow(np.tanh(result_state[-1]), vmin=-1, vmax=1)
    axs[i, 2].axis('off')

    axs[i, 0].text(-0.3, 0.5, f'$T={temp:.0f}$', transform=axs[i, 0].transAxes, va='center')

axs[0, 0].set_title('$t=0$')
axs[0, 1].set_title('$t=0.5t_{end}$')
axs[0, 2].set_title('$t=t_{end}$')

# place a color bar on the right side
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(axs[0, 0].imshow(np.tanh(result_state[0]), vmin=-1, vmax=1), cax=cbar_ax)

plt.show()
#plt.savefig('media/ising_grid.png', dpi=300)

#%%

# from plotting import plot_E_M_t

# plt.figure()

# plot_dict = {temp: (
#     sweep_results[sizes[-1]][temp][seeds[0]].state['time'],
#     sweep_results[sizes[-1]][temp][seeds[0]].state['energy'],
#     sweep_results[sizes[-1]][temp][seeds[0]].state['activity'],
# ) for temp in temperatures}

# plot_E_M_t(plot_dict)



#%%

