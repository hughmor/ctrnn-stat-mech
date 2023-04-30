#%%
import numpy as np
from context import sim_context

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=1.0, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = np.array([mu])
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def noise_process(temperature, noise_type=None, dtype=None):
    if dtype is None:
        dtype = sim_context.dtype
    dtype = np.dtype(dtype)

    num_eval = sim_context.num_eval
    timestep = sim_context.timestep
    shape = (sim_context.num_points, sim_context.num_points)
    if noise_type is None:
        noise = (np.random.normal(0, temperature, shape).astype(dtype) for _ in range(num_eval)) # does noise need to have its variance scaled by sqrt(timestep)?
    elif noise_type == 'gaussian':
        noise = (np.random.normal(0, 1.0, shape).astype(dtype)*temperature/np.sqrt(timestep) for _ in range(num_eval))
    elif noise_type == 'ou':
        noise_obj = OrnsteinUhlenbeckActionNoise(mu=np.zeros(shape, dtype), sigma=temperature, dt=timestep)
        noise = (noise_obj() for _ in range(num_eval))
    return noise

if __name__ == '__main__':
    # test the noise for varying values of theta
    import matplotlib.pyplot as plt

    theta_values = [0.0, 0.15, 1.0, 5.0, 10.0]
    for theta in theta_values:
        ou = OrnsteinUhlenbeckActionNoise(mu=0.0, sigma=1.0, theta=theta)
        ou.reset()
        x = [ou() for _ in range(1000)]
        plt.plot(x, label=f"theta={theta}")
    plt.legend()
    plt.show()


# %%
