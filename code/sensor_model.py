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
        # self._z_hit = 1.
        # self._z_short = 1. # 0.1
        # self._z_max = 0.5 # 0.1
        # self._z_rand = 5. # 100

        self._z_hit = 99/2/2.5/4  # 1.
        self._z_short = 198/4//2.5/4  # 1
        self._z_max = 49/2.5/4  # 0.5
        self._z_rand = 990/4  # 5

        self._sigma_hit = 100 # 15 # 50
        self._lambda_short = 0.01 # 0.05

        self._max_range = 1000 # 100 # 1000
        self._min_probability = 0.35
        self._subsampling = 2
        self._delta = 10

        self._norm_wts = 1.0

        self.occupancy_map = occupancy_map

    def beam_range_finder_model(self, z_t1_arr, x_t1, num_beams=180):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        stride = 180 // num_beams
        z_t1_arr = z_t1_arr[0:181:stride]
        z_t1_arr = z_t1_arr / 10.

        sensor_offset = 2.5 # sensor has 25 cm offset from car center
        theta = x_t1[2]
        x_sensor = x_t1.copy()
        x_sensor[:2] += (sensor_offset * np.array([np.cos(theta), np.sin(theta)]))

        z_t1_star = self.ray_casting(x_sensor, self.occupancy_map, num_beams)
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
        norm_term = norm.cdf(self._max_range, z_star, self._sigma_hit) - norm.cdf(0, z_star, self._sigma_hit)
        p_hit /= (norm_term + 1e-9)

        p_short = self._lambda_short * np.exp(-self._lambda_short * z)
        p_short /= (1 - np.exp(-self._lambda_short * z_star) + 1e-9)
        p_short[z > z_star] = 0

        p_max = np.ones_like(z) * (1. / self._delta)
        p_max[z < self._max_range - self._delta / 2] = 0

        p_rand = np.ones_like(z) * (1. / self._max_range)

        return p_hit, p_short, p_max, p_rand

    def ray_casting_old(self, x, occupancy_map, num_beams=180):
        """
        Implement ray casting algorithm.
        param[in] x: particle state belief [x, y, theta] at time t [world_frame]
        param[in] occupancy_map: 2D map
        param[in] num_beams[Optional]: Number of beams to cast. Interval: 180 / num_beams
        param[out] z_star: ground truth array for each beam
        """
        z_star = np.ones(num_beams)
        interval = 180 / num_beams
        for i in range(0, num_beams):
            theta = i * interval * 2 * np.pi # Convention - z* go from 0 degrees to 360 degrees
            # Could make it go from front CCW - theta = i*2*np.pi + x[2]
            x0, y0 = x[0:2]
            x1, y1 = x[0:2] + np.round(self._max_range * np.array([np.cos(theta), np.sin(theta)]))
            #print(x0, y0, x1, y1)
            z_star[i] = bresenham_line_search(x0, y0, x1, y1, occupancy_map, self._max_range)
            print(z_star[i])

        x0, y0 = x[0:2]
        thetas = np.arange(num_beams) * interval * 2 * np.pi
        x1 = x0 + z_star * np.cos(thetas)
        y1 = y0 - z_star * np.sin(thetas)

        # plot figure
        fig = plt.figure()
        plt.imshow(occupancy_map)
        plt.scatter(x0, y0, c='red')
        for i in range(num_beams):
            plt.plot((x0, x1[i]), (y0, y1[i]), c='yellow')

        plt.show()
        return z_star

    def ray_casting(self, x, occupancy_map, num_beams=180):
        z_star = np.ones(num_beams)
        thetas = np.arange(num_beams).astype(np.float) * (np.pi / num_beams)
        for i in range(num_beams):
            z_star[i] = self.ray_casting_one_direction(x, occupancy_map, thetas[i])
        return z_star

    def ray_casting_one_direction(self, x, occupancy_map, alpha):
        x0, y0, theta = x
        angle = (theta - np.pi/2 + alpha) % (2 * np.pi)
        if np.pi/2 < angle < np.pi*3/2:
            # beam point at negative x axis
            k = -np.tan(angle)
            for xi in range(int(np.round(x0)), 0, -1):
                yi = int(np.round((xi - x0) * k + y0))
                if 0 <= yi < occupancy_map.shape[0] and (occupancy_map[yi, xi] >= self._min_probability or occupancy_map[yi, xi] == -1):
                    return math.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)
            return math.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)
        elif angle < np.pi/2 or angle > np.pi*3/2:
            # beam point at positive x axis
            k = -np.tan(angle)
            for xi in range(int(np.round(x0)), occupancy_map.shape[1]):
                yi = int(np.round((xi - x0) * k + y0))
                if 0 <= yi < occupancy_map.shape[0] and (occupancy_map[yi, xi] >= self._min_probability or occupancy_map[yi, xi] == -1):
                    return math.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)
            return math.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)
        else:
            # angle == np.pi/2 or np.pi*3/2
            xi, yi = int(np.round(x0)), int(np.round(y0))
            y_offset = -1 if angle == np.pi/2 else 1
            while yi > 0 and yi < occupancy_map.shape[0]:
                if occupancy_map[yi, xi] >= self._min_probability or occupancy_map[yi, xi] == -1:
                    return math.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)
                yi += y_offset
            return math.sqrt((xi - x0) ** 2 + (yi - y0) ** 2)

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
    perform log sum trick to accumulate probability
    """
    return np.exp(np.sum(np.log(p)))


MIN_PROBABILITY = 0.35
def bresenham_line_search(x0, y0, x1, y1, occupancy_map, max_range):
    x0, y0, x1, y1 = np.round([x0, y0, x1, y1]).astype(int) # Casted to ints, may change later
    steep = np.abs(y1-y0) > abs(x1-x0)
    if steep:
        swap(x0, y0)
        swap(x1, y1)
    if x0 > x1:
        swap(x0, x1)
        swap(y0, y1)

    slope = (y1 - y0) / ( x1 - x0)
    derr = np.abs(slope)
    ystep = 1 - 2*(y0 > y1)
    err = 0.0
    y = y0

    for x in range(x0, x1):
        if steep:
            if occupancy_map[y, x] >= MIN_PROBABILITY: # IMPORTANT: is this how we determine there is an object there?
                # It's definitely not, since x/y are out of bounds
                return dist(x0, y0, x, y)
        else:
            if occupancy_map[x, y] >= MIN_PROBABILITY:
                return dist(x0, y0, x, y)
        err = err + derr
        if err >= 0.5:
            y = y + ystep
            err = err - 1.0

    return max_range


def swap(a, b):
    return b, a


def dist(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)


if __name__ == '__main__':
    np.random.seed(10008)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # generate a random particle, then sent out beams from that location
    h, w = occupancy_map.shape
    indices = np.where(occupancy_map.flatten() == 0)[0]
    ind = np.random.choice(indices, 1)[0]
    y, x = ind // w, ind % w
    theta = -np.pi/2
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


    # logfile = open('../data/log/robotdata1.log', 'r')
    # line = logfile.readlines()[0]
    #
    # meas_type = line[0]
    # meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
    #
    # odometry_robot = meas_vals[0:3]
    # time_stamp = meas_vals[-1]
    # if (meas_type == "L"):
    #     # [x, y, theta] coordinates of laser in odometry frame
    #     odometry_laser = meas_vals[3:6]
    #     # 180 range measurement values from single laser scan
    #     ranges = meas_vals[6:-1]
    #
    # print(f'Odometry_laser: {odometry_laser}, range 0: {ranges[0]}')
