import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
from motion_model import MotionModel


def test_motion_model():
    ### Generate test input
    np.random.seed(10008)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # generate a random particle, then sent out beams from that location
    h, w = occupancy_map.shape
    indices = np.where(occupancy_map.flatten() == 0)[0]
    ind = np.random.choice(indices, 1)[0]
    y, x = ind // w, ind % w
    theta = np.pi / 2
    X = np.array([[x, y, theta]])
    X[:, :2] *= 10

    # define motion control u, test on shift-only
    u_t1 = np.array([100, 200, theta])
    u_t2 = np.array([200, 300, theta])

    # motion model
    motion = MotionModel()
    X_t2 = motion.update_vectorized(u_t1, u_t2, X)

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


if __name__ == '__main__':
    test_motion_model()