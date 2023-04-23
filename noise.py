#%%
import numpy as np

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
