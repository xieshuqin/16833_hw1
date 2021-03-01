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

        # # First set of parameters that works
        # self._z_hit = 1000 / 1000 # 99 / 2 / 2.5 / 4  # 1.
        # self._z_short = 0.01 / 1000 # 2 * 198 / 4 / 2.5 / 4  # 1
        # self._z_max = 0.03 / 1000  # 49 / 4 / 4  # 0.5
        # self._z_rand = 12500 / 1000 # 990 / 4  # 5
        # self._sigma_hit = 250  # 400  # 15 # 50
        # self._lambda_short = 0.01  # 0.01  # 0.05

        # Second set of parameters that works, converge faster
        # self._z_hit = 1000 / 1000  # 99 / 2 / 2.5 / 4  # 1.
        # self._z_short = 0.01 / 100  # 2 * 198 / 4 / 2.5 / 4  # 1
        # self._z_max = 0.03 / 1000  # 49 / 4 / 4  # 0.5
        # self._z_rand = 12500 / 1000  # 990 / 4  # 5
        # self._sigma_hit = 100  # 400  # 15 # 50
        # self._lambda_short = 0.05  # 0.01  # 0.05

        # Third set of parameters that works
        self._z_hit = 1  # 99 / 2 / 2.5 / 4  # 1.
        self._z_short = 0.0001
        self._z_max = 0.0003
        self._z_rand = 12.5
        self._sigma_hit = 100
        self._lambda_short = 0.05

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 2
        self._delta = 50
        self._norm_wts = 1.0

        self.occupancy_map = occupancy_map

    def estimate_density(self, z, z_star):
        """
        param[in] z: 1D array of measurement [array of 180 values] at time t
        param[in] z_star: 1D array of ground truth location  [array of 180 values] at time t
        param[out] p_hit, p_short, p_max, p_rand: probability
        """
        p_hit = np.exp(-(z - z_star)**2 / (2*self._sigma_hit**2))
        p_hit /= np.sqrt(2 * np.pi * self._sigma_hit**2)

        p_short = self._lambda_short * np.exp(-self._lambda_short * z)
        p_short /= (1 - np.exp(-self._lambda_short * z_star) + 1e-9)
        p_short[z > z_star] = 0

        p_max = np.ones_like(z)
        p_max[z < self._max_range] = 0

        p_rand = np.ones_like(z) * (1. / self._max_range)

        return p_hit, p_short, p_max, p_rand

    def beam_range_finder_model(self, z_t1_arr, X_t1, num_beams=180):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] X_t1: M x 3, [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        stride = 180 // num_beams
        z_t1_arr = z_t1_arr[0:181:stride].copy()
        Z_t1_arr = np.repeat(z_t1_arr[None], X_t1.shape[0], axis=0)  # M x num_beams

        # compute sensor location, which shifts 25cm from car center
        sensor_offset = 25
        theta = X_t1[:, 2]
        X_sensor = X_t1.copy()
        X_sensor[:, 0] += (sensor_offset * np.cos(theta))
        X_sensor[:, 1] += (sensor_offset * np.sin(theta))

        Z_t1_star = self.ray_casting(X_sensor, num_beams)
        p_hit, p_short, p_max, p_rand = self.estimate_density(Z_t1_arr, Z_t1_star)
        prob_zt1_arr = self._z_hit * p_hit + self._z_short * p_short + self._z_rand * p_rand + self._z_max * p_max
        log_prob_zt1 = log_sum(prob_zt1_arr)
        return log_prob_zt1

    def ray_casting(self, X_cm, num_beams=180):
        """
        Input: X_cm: [M, 3], particle states at centimeter
               num_beams: int
        Output:
               Z_star: [M x num_beams]: ground truth obstacle at centimeter
        """
        # convert centimeter to indices
        X = X_cm.copy()
        X[:, :2] /= 10

        # generate num_beams for each particle, compute their x, y, theta
        X = np.repeat(X[:, None, :], num_beams, axis=1)  # M x num_beams x 3
        stride = 180 // num_beams
        scan_angles = np.arange(0, 180, stride) * (np.pi / 180)
        X[..., 2] += scan_angles - np.pi/2
        X[..., 2] %= (2 * np.pi)
        X = X.reshape(-1, 3)  # M*num_beams x 3, [x, y, theta]

        # main loop
        dist = np.zeros((X.shape[0],))  # distance to emitters
        X_unhit = X
        dist_unhit = dist
        unhit_inds = np.arange(len(X_unhit))
        while len(unhit_inds) > 0:
            # traverse for one step and check hit
            X_unhit, dist_unhit = self.traverse_one_step(X_unhit, dist_unhit)
            hit_mask = self.check_hit(X_unhit, dist_unhit)
            # update variables if any ray is hit
            if np.any(hit_mask):
                dist[unhit_inds[hit_mask]] = dist_unhit[hit_mask]
                unhit_mask = ~hit_mask
                unhit_inds = unhit_inds[unhit_mask]
                X_unhit = X_unhit[unhit_mask]
                dist_unhit = dist_unhit[unhit_mask]

        Z_star = dist.reshape(-1, num_beams)
        Z_star *= 10.  # convert to centimeter
        return Z_star

    def traverse_one_step(self, X, dist):
        """
        Traverse along a ray for one step.
        """
        x, y, theta = X[:, 0], X[:, 1], X[:, 2]
        slope = np.tan(theta)
        STEPSIZE = 1

        # angles between pi/4 ~ 3pi/4, increment y and compute x
        indices = np.where((np.pi/4 <= theta) & (theta <= 3*np.pi/4))[0]
        y_step = 1. * STEPSIZE
        y[indices] += y_step
        x[indices] += (y_step / slope[indices])
        dist[indices] += np.sqrt(y_step**2 + (y_step / slope[indices])**2)

        # angles between 5pi/4 ~ 7pi/4, decrement y and compute x
        indices = np.where((5*np.pi/4 <= theta) & (theta <= 7*np.pi/4))[0]
        y_step = -1. * STEPSIZE
        y[indices] += y_step
        x[indices] += (y_step / slope[indices])
        dist[indices] += np.sqrt(y_step**2 + (y_step / slope[indices])**2)

        # angles between 0 ~ pi/4 and 7pi/4 ~ 2pi, increment x and compute y
        indices = np.where((theta < np.pi/4) | (theta > 7*np.pi/4))[0]
        x_step = 1. * STEPSIZE
        x[indices] += x_step
        y[indices] += (x_step * slope[indices])
        dist[indices] += np.sqrt(x_step**2 + (x_step * slope[indices])**2)

        # angles between 3pi/4 ~ 5pi/4, decrement x and compute y
        indices = np.where((3*np.pi/4 < theta) & (theta < 5*np.pi/4))[0]
        x_step = -1. * STEPSIZE
        x[indices] += x_step
        y[indices] += (x_step * slope[indices])
        dist[indices] += np.sqrt(x_step**2 + (x_step * slope[indices])**2)

        return X, dist

    def check_hit(self, X, dist):
        """
        Check if a ray hits obstacle or exceeds max range
        param X: M x 3
        param: dist: M
        """
        H, W = self.occupancy_map.shape
        x, y = np.split(np.round(X[:, :2]).astype(np.int), 2, axis=1)
        mask = ((0 <= x) & (x < W)) & ((0 <= y) & (y < H))
        indices = []

        # Consider out-of-boundary particles as found
        oob_indices = np.where(~mask)[0]
        indices.append(oob_indices)

        # For particles within boundary, consider hitting obstacle as found
        ib_indices = np.where(mask)[0]
        x, y = x[ib_indices], y[ib_indices]
        map = self.occupancy_map
        hit_mask = (map[y, x] >= self._min_probability) | (map[y, x] == -1)
        hit_indices = ib_indices[np.where(hit_mask)[0]]
        indices.append(hit_indices)

        # Consider out-of-max-range as found
        oor_indices = np.where(dist > (self._max_range / 10.))[0]
        dist[oor_indices] = (self._max_range / 10.)
        indices.append(oor_indices)

        indices = np.concatenate(indices)
        mask = np.zeros((X.shape[0],), dtype=np.bool)
        mask[indices] = True
        return mask


def log_sum(p):
    return np.sum(np.log(p), axis=1)


if __name__ == '__main__':
    np.random.seed(10008)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # generate a random particle, then sent out beams from that location
    h, w = occupancy_map.shape
    indices = np.where(occupancy_map.flatten() == 0)[0]
    ind = np.random.choice(indices, 1)[0]
    y, x = ind // w, ind % w
    theta = np.pi/2
    angle = np.pi*(95. / 180)

    sensor = SensorModel(occupancy_map)
    z_t_star = sensor.ray_casting_one_direction((x, y, theta), occupancy_map, angle)

    x0, y0 = x, y
    angle = theta + angle - np.pi/2
    x1 = x0 + z_t_star * np.cos(angle)
    y1 = y0 - z_t_star * np.sin(angle)

    # plot figure
    fig = plt.figure()
    plt.imshow(occupancy_map)
    plt.scatter(x0, y0, c='red')
    plt.scatter(x1, y1, c='yellow')
    print(f'(x0, y0): ({x0}, {y0}), (x1, y1): ({x1}, {y1})')
    plt.plot((x0, x1), (y0, y1), color='yellow')

    plt.show()
    print(z_t_star)
