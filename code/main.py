'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling, log_prob_to_prob

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

from sklearn.cluster import DBSCAN

# For debug purpose, set a fix random seed
np.random.seed(10000)


def visualize_map(occupancy_map):
    fig = plt.figure(1)
    # ax = plt.subplot(121)
    # mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    fig = plt.figure(1)
    # ax = plt.subplot(121)
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    # plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.01)
    scat.remove()


def visualize_rays(X_bar, sensor_model):
    x_peak = X_bar[np.argmax(X_bar[:, 3])]
    x, y, theta, weight = x_peak
    x0, y0 = x, y
    angle = np.arange(180) * np.pi / 180
    angle = theta + angle - np.pi/2

    z_t_star = sensor_model.ray_casting(np.array([x, y, theta]), sensor_model.occupancy_map, 180)
    x1 = x0 + z_t_star * np.cos(angle)
    y1 = y0 - z_t_star * np.sin(angle)

    x0 /= 10
    y0 /= 10
    x1 /= 10
    y1 /= 10

    # fig = plt.figure(2)
    fig = plt.figure(1)
    ax = plt.subplot(122)
    plt.cla()
    plt.imshow(sensor_model.occupancy_map)
    plt.scatter(x1, y1, c='y', marker='o')
    plt.scatter(x0, y0, c='r', marker='x')
    plt.pause(0.01)
    # plt.show()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
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


def adaptively_choose_particle(X):
    cluster = DBSCAN(eps=10, min_samples=2).fit(X[:, :2])
    labels = cluster.labels_
    X_out = []
    for unique_label in np.unique(labels):
        if unique_label == -1:
            continue
        mask = labels == unique_label
        X_cluster = X[mask]
        prob = log_prob_to_prob(X_cluster[:, -1])
        indices = np.random.choice(len(X_cluster), min(len(X_cluster), 100), replace=False, p=prob)
        X_out.append(X_cluster[indices])
    return np.concatenate(X_out, axis=0)


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=10000, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only send out 10 beams')
    parser.add_argument('--adaptive', action='store_true', help='Adaptively select particles or not')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    with open(src_path_log, 'r') as f:
        logfile = f.readlines()

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    num_beams = 10 if args.debug else 18
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)
        visualize_timestep(X_bar, 0, args.output)

    first_time_idx = True
    total_time, cnt = 0., 0.
    for time_idx, line in enumerate(logfile):
        tic = time.time()
        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        if meas_type == "L":
            u_t1 = odometry_robot

            # Update motion
            X_t1 = motion_model.update(u_t0, u_t1, X_bar[:, 0:3])

            # Correction step
            z_t = ranges
            W_t = sensor_model.beam_range_finder_model(z_t, X_t1, num_beams)
            X_bar_new = np.hstack((X_t1, W_t[..., None]))

            u_t0 = u_t1

            # Resampling step
            X_bar_new = resampler.low_variance_sampler(X_bar_new)

            # DBSCAN to sample subsets of particles
            if args.adaptive:
                X_bar_new = adaptively_choose_particle(X_bar_new)
            X_bar = X_bar_new

        if meas_type == "L":
            toc = time.time()
            total_time += (toc - tic)
            cnt += 1
            # print(f'Running time: {(1./(toc - tic)):.2f} fps, {toc-tic}s')

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)

    avg_time = total_time / cnt
    print(f'Average FPS: {(1./avg_time):.2f} fps')