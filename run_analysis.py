import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

def animate_grid(sim_results, filename):
    # sim results is a list of NxN arrays representing the state of the grid at each time step
    # using imshow to animate the grid
    # plot two images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    im1 = ax[0].imshow(sim_results[0], cmap='gray', vmin=0, vmax=1)
    im2 = ax[1].imshow(np.tanh(sim_results[0]), cmap='gray', vmin=0, vmax=1)

    ax[0].set_title('State')
    ax[1].set_title('Activation')

    def animate(i):
        im1.set_data(sim_results[i])
        im2.set_data(np.tanh(sim_results[i]))
        return im1, im2

    anim = animation.FuncAnimation(fig, animate, frames=len(sim_results), interval=20, blit=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed
    anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])

def torus(nx = 100, ny=100, r=1, R=2):
    theta = np.linspace(0, 2.*np.pi, ny)
    phi = np.linspace(0, 2.*np.pi, nx)
    theta, phi = np.meshgrid(theta, phi)
    c, a = R, r
    x = (c + a*np.cos(theta)) * np.cos(phi)
    y = (c + a*np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    return x, y, z

def animate_torus(sim_results, filename):
    # same as animate_grid, but show the results on a torus
    # in this function, only plot the state, not the activation
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    first_image = sim_results[0]
    x, y, z = torus(nx=first_image.shape[0], ny=first_image.shape[1])

    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-3,3)

    # turn image value into rgba after normalizing to [0,1]
    first_image = (first_image - first_image.min()) / (first_image.max() - first_image.min())
    first_image = np.dstack((first_image, first_image, first_image, np.ones(first_image.shape)))

    ax.plot_surface(x, y, z, rstride=5, cstride=5, facecolors=first_image) # color='k', edgecolors='w')
    ax.view_init(36, 26)

    def animate(i):
        ax.clear()
        ax.set_zlim(-3,3)
        image = sim_results[i]
        image = (image - image.min()) / (image.max() - image.min())
        image = np.dstack((image, image, image, np.ones(image.shape)))
        artists = ax.plot_surface(x, y, z, rstride=5, cstride=5, facecolors=image)
        ax.view_init(36, 26)
        return artists,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(sim_results), interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed
    anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])


def main():
    # load the results
    filename = 'grid_randombias_0_01_noise'
    results = np.load(f'data/{filename}.npy')

    # animate the results
    animate_grid(results, f'media/{filename}.mp4')
    #animate_torus(results, f'media/{filename}_torus.mp4')

if __name__ == '__main__':
    main()