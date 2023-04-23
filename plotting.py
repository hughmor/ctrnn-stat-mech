import numpy as np
import matplotlib.pyplot as plt
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