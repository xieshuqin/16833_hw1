'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self, occupancy=None):
        """
        TODO : Initialize resampling process parameters here
        """
        self.w_slow = 1.
        self.w_fast = 1.
        self.alpha_slow = 0.01
        self.alpha_fast = 0.1 # alpha_fast >> alpha_slow > 0
        self.occupancy = occupancy

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        prob = log_prob_to_prob(X_bar[:, -1])
        indices = np.random.choice(num_particles, num_particles, replace=True, p=prob)
        X_bar_resampled = X_bar[indices]
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = []
        num_particles = X_bar.shape[0]
        W = log_prob_to_prob(X_bar[:, -1])
        r = np.random.rand(1)[0] * (1. / num_particles)
        i = 0
        c = W[0]
        for m in range(0, num_particles):
            U = r + m * 1. / num_particles
            while U > c:
                i += 1
                c += W[i]
            X_bar_resampled.append(X_bar[i])
        X_bar_resampled = np.stack(X_bar_resampled)
        return X_bar_resampled

    def random_particle_sampler(self, X_bar):
        num_particles = X_bar.shape[0]
        W = log_prob_to_prob(X_bar[:, -1])
        w_avg = W.mean()
        self.w_slow += self.alpha_slow * (w_avg - self.w_slow)
        self.w_fast += self.alpha_fast * (w_avg - self.w_fast)

        prob = np.random.uniform(size=num_particles)
        mask = prob < max(0, 1 - self.w_fast / self.w_slow)

        # sample from existing particles
        sample_existed = X_bar[np.random.choice(num_particles, num_particles, replace=True, p=W)][~mask]

        # sample from random freespace
        sample_random = init_particles_freespace(num_particles, self.occupancy)[mask]

        X_bar_resampled = np.stack([sample_existed, sample_random], axis=0)
        return X_bar_resampled


def log_prob_to_prob(log_prob):
    max_value = np.max(log_prob)
    prob = np.exp(log_prob - max_value)
    prob /= np.sum(prob)
    return prob


def init_particles_freespace(num_particles, occupancy_map):
    # initialize [x, y, theta] positions in world_frame for all particles
    """
    This version converges faster than init_particles_random
    """
    MIN_PROBABILITY = 0.35
    # y, x = np.where((occupancy_map < MIN_PROBABILITY) & (occupancy_map != -1))
    y, x = np.where(occupancy_map == 0)
    indices = np.random.choice(len(y), num_particles, replace=False)
    y0_vals = y[indices].astype(np.float) * 10.
    x0_vals = x[indices].astype(np.float) * 10.
    theta0_vals = np.random.uniform(-np.pi, np.pi, num_particles)

    w0_vals = np.ones((num_particles,), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.stack([x0_vals, y0_vals, theta0_vals, w0_vals], axis=1)
    return X_bar_init
