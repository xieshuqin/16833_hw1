'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0001 # 0.0002
        self._alpha2 = 0.0001 # 0.0004
        self._alpha3 = 0.01 # 0.01
        self._alpha4 = 0.01 # 0.01

    def update(self, u_t0_raw, u_t1_raw, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief M x 3, [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief M x 3, [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        u_t0 = u_t0_raw.copy()
        u_t1 = u_t1_raw.copy()

        delta_rot_1 = np.arctan2(u_t1[1]-u_t0[1],u_t1[0]-u_t0[0])-u_t0[2]
        delta_trans = np.sqrt((u_t0[0]-u_t1[0])**2+(u_t0[1]-u_t1[1])**2)
        delta_rot_2 = u_t1[2]-u_t0[2]-delta_rot_1

        M = x_t0.shape[0]
        delta_rot_1_hat = delta_rot_1-np.random.normal(0.0, np.sqrt(np.abs(self._alpha1*(delta_rot_1**2)+self._alpha2*(delta_trans**2))), M)
        delta_trans_hat = delta_trans-np.random.normal(0.0, np.sqrt(np.abs(self._alpha3*(delta_trans**2)+self._alpha4*(delta_rot_1**2+delta_rot_2**2))), M)
        delta_rot_2_hat = delta_rot_2-np.random.normal(0.0, np.sqrt(np.abs(self._alpha1*(delta_rot_2**2)+self._alpha2*(delta_trans**2))), M)

        x_prime = x_t0[:, 0] + delta_trans_hat*np.cos(x_t0[:, 2]+delta_rot_1_hat)
        y_prime = x_t0[:, 1] + delta_trans_hat*np.sin(x_t0[:, 2]+delta_rot_1_hat)
        theta_prime = x_t0[:, 2] + delta_rot_1_hat + delta_rot_2_hat

        return np.stack([x_prime, y_prime, theta_prime], axis=1)
