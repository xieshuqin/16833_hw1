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
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        prob = X_bar[:, -1] / np.sum(X_bar[:, -1])
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
        W = X_bar[:, -1] / np.sum(X_bar[:, -1])
        r = np.random.rand(1)[0] * (1./num_particles)
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


if __name__ == '__main__':
    resampler = Resampling()
    X_bar = np.random.rand(20, 4)
    sample1 = resampler.multinomial_sampler(X_bar)
    sample2 = resampler.low_variance_sampler(X_bar)
    print(f'X_bar: {X_bar}')
    print(f'sample2: {sample2}')
    print(f'sample1: {sample1}')