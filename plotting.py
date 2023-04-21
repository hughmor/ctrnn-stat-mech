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

def plot_grid(sim_results, filename=None, time_idx=-1, title=None):
    # this function sets up the plot which will either be static or passed to an animation function to animate

    # sim results is a list of NxN arrays representing the state of the grid at each time step
    # plot two images side by side with no axis labels or ticks
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    vmin = min([np.min(frame) for frame in sim_results])
    vmax = max([np.max(frame) for frame in sim_results])
    im1 = ax[0].imshow(sim_results[time_idx], cmap='copper', vmin=vmin, vmax=vmax)
    im2 = ax[1].imshow(np.tanh(sim_results[time_idx]), cmap='cool', vmin=-1, vmax=1)
    ax[0].axis('off')
    ax[1].axis('off')

    # put one colorbar on the left and one on the right
    # make the colorbars the same height as the image instead of the whole subplot
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    # add some separation between the subplots
    fig.subplots_adjust(wspace=0.3)

    # set the titles
    ax[0].set_title('State')
    ax[1].set_title('Activation')

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig, ax, im1, im2


def animate_grid(sim_results, filename=None, title=None, embed='js'):
    fig, ax, im1, im2 = plot_grid(sim_results, time_idx=0, title=title)

    def animate(i):
        im1.set_data(sim_results[i])
        im2.set_data(np.tanh(sim_results[i]))
        return im1, im2
    anim = animation.FuncAnimation(fig, animate, frames=len(sim_results), interval=100, blit=True)
    
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