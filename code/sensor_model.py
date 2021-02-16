'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 2
        self._delta = 2

        self._norm_wts = 1.0

        self.occupancy_map = occupancy_map

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        z_t1_star = self.ray_casting(x_t1, self.occupancy_map)
        p_hit, p_short, p_max, p_rand = self.estimate_density(z_t1_arr, z_t1_star)
        prob_zt1_arr = self._z_hit * p_hit + self._z_short * p_short + self._z_rand * p_rand + self._z_max * p_max
        prob_zt1 = log_sum(prob_zt1_arr)
        return prob_zt1

    def estimate_density(self, z, z_star):
        """
        param[in] z: 1D array of measurement [array of 180 values] at time t
        param[in] z_star: 1D array of ground truth location  [array of 180 values] at time t
        param[out] p_hit, p_short, p_max, p_rand: probability
        """
        p_hit = np.exp(-(z - z_star)**2 / (2*self._sigma_hit))
        p_hit /= np.sqrt(2 * np.pi * self._sigma_hit)
        p_hit /= 1 # TODO: Add a normalize term here

        p_short = self._lambda_short * np.exp(-self._lambda_short * z)
        p_short /= (1 - np.exp(-self._lambda_short * z_star))
        p_short[z > z_star] = 0

        p_max = np.ones_like(z) * (1. / self._delta)
        p_max[z < self._max_range - self._delta / 2] = 0

        p_rand = np.ones_like(z) * (1. / self._max_range)

        return p_hit, p_short, p_max, p_rand

    def ray_casting(self, x, occupancy_map, num_beams=180):
        """
        Implement ray casting algorithm.
        param[in] x: particle state belief [x, y, theta] at time t [world_frame]
        param[in] occupancy_map: 2D map
        param[in] num_beams[Optional]: Number of beams to cast. Interval: 180 / num_beams
        param[out] z_star: ground truth array for each beam
        """
        z_star = np.ones(num_beams)
        return z_star

    def learn_intrinsic_parameters(self, Z, X, num_beams=8):
        converge = False
        old_params = np.array(
            [self._z_hit, self._z_short, self._z_max, self._z_rand, self._sigma_hit, self._lambda_short])
        num_iterations = 0
        while not converge:
            # compute Z_star for each position, only use num_beams measurement for each location.
            indices = np.arange(0, 180, 180 // num_beams)
            Z_prime = Z[:, indices]
            Z_star = []
            for z_i, x_i in zip(Z_prime, X):
                z_i_star = self.ray_casting(x_i, self.occupancy_map, num_beams)
                Z_star.append(z_i_star)
            Z_star = np.concatenate(Z_star)
            Z_prime = Z_prime.flatten()

            # compute error
            p_hit, p_short, p_max, p_rand = self.estimate_density(Z_prime, Z_star)
            eta = 1. / (p_hit + p_short + p_max + p_rand)
            e_hit = eta * p_hit
            e_short = eta * p_short
            e_max = eta * p_max
            e_rand = eta * p_rand

            # update parameters
            Z_size = len(Z_prime)
            self._z_hit = np.sum(e_hit) / Z_size
            self._z_short = np.sum(e_short) / Z_size
            self._z_max = np.sum(e_max) / Z_size
            self._z_rand = np.sum(e_rand) / Z_size
            self._sigma_hit = np.sqrt(np.sum(e_hit * (Z_prime - Z_star)**2) / (e_hit * Z_size))
            self._lambda_short = np.sum(e_short) / np.sum(e_short * Z_prime)

            # check converge
            new_params = np.array(
                [self._z_hit, self._z_short, self._z_max, self._z_rand, self._sigma_hit, self._lambda_short])
            converge = np.allclose(old_params, new_params, rtol=1e-2)
            num_iterations += 1
            print(f'iteration {num_iterations}: old params {old_params}, new params {new_params}')
            old_params = new_params

def log_sum(p):
    """
    perform log sum trick to accumualte probability
    """
    return np.exp(np.sum(np.log(p)))