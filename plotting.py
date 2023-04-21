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

def plot_grid(sim_results, filename=None, time_idx=-1, time=None, title=None):
    # this function sets up the plot which will either be static or passed to an animation function to animate

    # sim results is a list of NxN arrays representing the state of the grid at each time step
    # plot two images side by side with no axis labels or ticks
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    vmin = min([np.min(frame) for frame in sim_results])
    vmax = max([np.max(frame) for frame in sim_results])
    im1 = ax[0].imshow(sim_results[time_idx], cmap='plasma', vmin=vmin, vmax=vmax)
    im2 = ax[1].imshow(np.tanh(sim_results[time_idx]), cmap='plasma', vmin=-1, vmax=1)
    ax[0].axis('off')
    ax[1].axis('off')

    if time is not None:
        txt = ax[0].text(0.11, 1.03, f'$t={time[time_idx]:.2f}$', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    else:
        txt = None

    # put one colorbar on the left and one on the right
    # make the colorbars the same height as the image instead of the whole subplot
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    # add some separation between the subplots
    fig.subplots_adjust(wspace=0.15)

    # set the titles
    ax[0].set_title('State ($s$)\n')
    ax[1].set_title('Activation ($\\sigma(s)$)\n')

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig, ax, (im1, im2, txt)


def animate_grid(sim_results, time=None, filename=None, title=None, embed='js', plot_every=1):
    fig, ax, artists = plot_grid(sim_results, time_idx=0, title=title, time=time)

    im1, im2, txt = artists

    def animate(i):
        im1.set_data(sim_results[i])
        im2.set_data(np.tanh(sim_results[i]))
        if time is not None:
            txt.set_text(f'$t={time[i]:.2f}$')
            return im1, im2, txt
        else:
            return im1, im2
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