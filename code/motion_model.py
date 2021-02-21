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
        self._alpha1 = 100 # 0.0002
        self._alpha2 = 100 # 0.0004
        self._alpha3 = 100 # 0.01
        self._alpha4 = 100 # 0.01


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        delta_rot_1 = np.arctan2(u_t1[1]-u_t0[1],u_t1[0]-u_t0[0])-u_t0[2]
        delta_trans = np.sqrt((u_t0[0]-u_t1[0])**2+(u_t0[1]-u_t1[1])**2)
        delta_rot_2 = u_t1[2]-u_t0[2]-delta_rot_1

        if delta_rot_2 > np.pi or delta_rot_2 < -np.pi:
            mult = int(delta_rot_2/np.pi)
            delta_rot_2 = (delta_rot_2-mult*np.pi)

        delta_rot_1_hat = delta_rot_1-np.random.normal(0.0, np.sqrt(np.abs(self._alpha1*delta_rot_1+self._alpha2*delta_trans)))
        delta_trans_hat = delta_trans-np.random.normal(0.0, np.sqrt(np.abs(self._alpha3*delta_trans+self._alpha4*(delta_rot_1+delta_rot_2))))
        delta_rot_2_hat = delta_rot_2-np.random.normal(0.0, np.sqrt(np.abs(self._alpha1*delta_rot_2+self._alpha2*delta_trans))) 

        x_prime = x_t0[0] + delta_trans_hat*np.cos(x_t0[2]+delta_rot_1_hat)
        y_prime = x_t0[1] + delta_trans_hat*np.sin(x_t0[2]+delta_rot_1_hat)
        theta_prime = x_t0[2] + delta_rot_1_hat + delta_rot_2_hat

        return np.array([x_prime, y_prime, theta_prime]).T
