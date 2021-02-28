import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
from motion_model import MotionModel

from main import visualize_map, visualize_timestep


def test_motion_model():
    ### Generate test input
    np.random.seed(10000)
    np.set_printoptions(precision=4, suppress=True)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # # generate a random particle, then sent out beams from that location
    # h, w = occupancy_map.shape
    # indices = np.where(occupancy_map.flatten() == 0)[0]
    # ind = np.random.choice(indices, 1)[0]
    # y, x = ind // w, ind % w
    # theta = np.pi / 2
    # X = np.array([[x, y, theta]])
    # X[:, :2] *= 10

    h, w = occupancy_map.shape
    flipped_occupancy_map = occupancy_map[::-1, :]
    y, x = np.where(flipped_occupancy_map == 0)
    valid_indices = np.where((350 <= y) & (y <= 450) & (350 <= x) & (x <= 450))[0]
    ind = indices = np.random.choice(valid_indices, 1)[0]
    y, x = y[ind], x[ind]
    # y, x = ind // w, ind % w
    theta = np.pi / 4
    X = np.array([[x, y, theta]])
    X[:, :2] *= 10
    print(X)

    # define motion control u, test on shift-only
    theta1 = np.pi / 2
    u_t1 = np.array([1049.833130, 28.126379, -2.664653])
    u_t2 = np.array([1048.641235, 27.560303, -2.659024])

    # motion model
    motion = MotionModel()
    X_t2 = motion.update(u_t1, u_t2, X)
    print(X_t2)

    print(f'X: {X}, X_t2: {X_t2}')

    # plot
    plt.figure(1)
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])

    x_loc = X[:, 0] / 10.
    y_loc = X[:, 1] / 10.
    plt.scatter(x_loc, y_loc, c='r', marker='o')

    x_loc = X_t2[:, 0] / 10.
    y_loc = X_t2[:, 1] / 10.
    plt.scatter(x_loc, y_loc, c='y', marker='o')
    plt.show()


def test_ground_truth_motion():
    # Read map
    np.random.seed(10000)
    np.set_printoptions(precision=4, suppress=True)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # specify ground truth location
    X = np.array([[4200, 4000, np.pi*(170/180.)]])
    motion_model = MotionModel()

    with open('../data/log/robotdata1.log', 'r') as f:
        logfile = f.readlines()

    visualize_map(occupancy_map)
    visualize_timestep(X, 0, '../tmp')

    first_time_idx = True
    for time_idx, line in enumerate(logfile):
        meas_type = line[0]
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        # print("Processing time step {} at time {}s".format(
        #     time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        if meas_type == "O":
            continue
        u_t1 = odometry_robot

        # Update motion
        X = motion_model.update(u_t0, u_t1, X)
        u_t0 = u_t1

        visualize_timestep(X, time_idx, '../tmp')


if __name__ == '__main__':
    test_ground_truth_motion()