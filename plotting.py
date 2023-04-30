import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from IPython.display import HTML

cmaps = [
    'viridis',
    'plasma',
    'inferno',
    'magma',
    'cividis',
    'gray',
    'copper',
    'hot',
    'afmhot',
    'cool',
]

def plot_E_M_t(states):
    fig, (m_ax, e_ax) = plt.subplots(2, 1, sharex=True)
    temps = list(states.keys())
    maxtemp = max(temps)
    mintemp = min(temps)
    for temp, (t,e,m) in states.items():
        # color the lines by temperature
        cmap = plt.get_cmap('viridis')
        temp_norm = (temp - mintemp) / (maxtemp - mintemp)

        m_ax.plot(t, m, label=f'T={temp:.1f}', color=cmap(temp_norm))
        e_ax.plot(t, e, color=cmap(temp_norm)) # , label=f'T={temp:.1f}' removed because it's too many labels

    # make space for cbar to the right
    fig.subplots_adjust(right=0.95)
    # add colorbar for temperature
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    norm = plt.Normalize(mintemp, maxtemp)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Temperature ($T$)')


    m_ax.set_ylabel('Activity ($A(t)$)')
    e_ax.set_ylabel('Energy ($E(t)$)')
    e_ax.set_xlabel('Time ($t$)')
    plt.show()

def plot_M_t(states):
    plt.figure()
    temps = list(states.keys())
    maxtemp = max(temps)
    mintemp = min(temps)
    # add line through y=0
    plt.axhline(0, color='black', linestyle='-', lw=1)

    for sim_result in states:
        temp = sim_result.metadata['temperature']
        m = sim_result.state['activity']
        t = sim_result.state['time']
        plt.plot(t, m, label=f'T={temp:.1f}')

    plt.xlabel('Time ($t$)')
    plt.ylabel('Activity ($A(t)$)')
    plt.legend()

def plot_s_t(x,y,sim_state,fig=None):
    # if x and y are ints, plot the state at that point over time
    # if x and y are lists, plot the state at those points over time
    
    if fig is None:
        fig = plt.figure()
    if isinstance(x, int) and isinstance(y, int):
        x = [x]
        y = [y]
    elif isinstance(x, list) and isinstance(y, list):
        pass
    else:
        raise ValueError('x and y must both be ints or lists')
    
    grid = np.array(sim_state.state['grid'])
    for x_, y_ in zip(x,y):
        plt.plot(sim_state.state['time'], grid[:,x_,y_], label=f'({x_},{y_})')
    
    if fig is None: 
        plt.xlabel('Time ($t$)')
        plt.ylabel('State ($s(t)$)')
        plt.legend()

def plot_sampled_s_t(states, N=5):
    sim_result = states[-1]
    grid = np.array(sim_result.state['grid'])

    plt.figure()
    # pick a few random points and plot over time
    for _ in range(N):
        x = np.random.randint(0, grid.shape[1])
        y = np.random.randint(0, grid.shape[2])
        plot_s_t(x,y,sim_result,fig=plt.gcf())
    
    plt.xlabel('Time ($t$)')
    plt.ylabel('State ($s(t)$)')
    plt.legend()


def plot_E_t(states):
    plt.figure()

    for sim_result in states:
        temp = sim_result.metadata['temperature']
        e = sim_result.state['energy']
        t = sim_result.state['time']
        plt.plot(t, e, label=f'T={temp:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Energy ($E(t)$)')
    plt.legend()


def plot_grid_T(tiles, time_idx=-1):
    n = len(tiles)
    N = round(np.sqrt(n))
    fig, ax = plt.subplots(N, N+1, figsize=(10,10))
    for i, tile in enumerate(tiles):
        grid = tile.state['grid']
        temp = tile.metadata['temperature']
        ax[i//(N+1), i%(N+1)].imshow(grid[time_idx], cmap='plasma', vmin=-1, vmax=1)
        ax[i//(N+1), i%(N+1)].axis('off')
        ax[i//(N+1), i%(N+1)].set_title(f'T={temp:.1f}')
    plt.show()

def plot_grid(sim_results, plot_idx=None, filename=None, time_idx=-1, time=None, title=None):
    # this function sets up the plot which will either be static or passed to an animation function to animate

    # sim results is a list of NxN arrays representing the state of the grid at each time step
    # plot two images side by side with no axis labels or ticks
    vmin = min([np.min(frame) for frame in sim_results])
    vmax = max([np.max(frame) for frame in sim_results])
    if plot_idx is None:
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        state_axis = ax[0]
        activation_axis = ax[1]
    elif plot_idx == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        state_axis = ax
        activation_axis = None
    elif plot_idx == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        state_axis = None
        activation_axis = ax
    
    if state_axis is not None:
        im1 = state_axis.imshow(sim_results[time_idx], cmap='plasma', vmin=vmin, vmax=vmax)
        state_axis.axis('off')
        state_axis.set_title('State ($s$)\n')
    else:
        im1 = None
    if activation_axis is not None:
        im2 = activation_axis.imshow(np.tanh(sim_results[time_idx]), cmap='plasma', vmin=-1, vmax=1)
        activation_axis.axis('off')
        activation_axis.set_title('Activation ($\\sigma(s)$)\n')
    else:
        im2 = None

    if time is not None:
        if state_axis is not None:
            txt = state_axis.text(0.11, 1.03, f'$t={time[time_idx]:.2f}$', horizontalalignment='center', verticalalignment='center', transform=state_axis.transAxes)
        else:
            txt = activation_axis.text(0.11, 1.03, f'$t={time[time_idx]:.2f}$', horizontalalignment='center', verticalalignment='center', transform=activation_axis.transAxes)
    else:
        txt = None

    # put one colorbar on the left and one on the right
    # make the colorbars the same height as the image instead of the whole subplot
    if state_axis is not None:
        fig.colorbar(im1, ax=state_axis, fraction=0.046, pad=0.04)
    if activation_axis is not None:
        fig.colorbar(im2, ax=activation_axis, fraction=0.046, pad=0.04)

    if plot_idx is None:
        # add some separation between the subplots
        fig.subplots_adjust(wspace=0.15)

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig, ax, (im1, im2, txt)


def animate_grid(sim_results, time=None, filename=None, title=None, plot_idx=None, embed='js', plot_every=1):
    fig, ax, artists = plot_grid(sim_results, plot_idx=plot_idx, time_idx=0, title=title, time=time)

    im1, im2, txt = artists

    def animate(i):
        if im1 is not None:
            im1.set_data(sim_results[i])
        if im2 is not None:
            im2.set_data(np.tanh(sim_results[i]))
        if time is not None:
            txt.set_text(f'$t={time[i]:.2f}$')
        
        def update():
            if im1 is not None:
                yield im1
            if im2 is not None:
                yield im2
            if txt is not None:
                yield txt
        
        return update()

    anim = animation.FuncAnimation(fig, animate, frames=range(0,len(sim_results),plot_every), interval=100, blit=True)
    
    if filename is None: # running from Jupyter notebook
        if embed == 'js':
            return HTML(anim.to_jshtml())
        elif embed == 'html5':
            return HTML(anim.to_html5_video())
        else:
            raise ValueError(f'embed must be "js" or "html5", got {embed}')
    else: # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed
        if not filename.endswith('.mp4'): filename += '.mp4'
        anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
        print(f'Animation saved to {filename}')