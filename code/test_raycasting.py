import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
from sensor_model import SensorModel

def test_raycasting_vectorize():

    np.random.seed(10008)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # generate a random particle, then sent out beams from that location
    h, w = occupancy_map.shape
    indices = np.where(occupancy_map.flatten() == 0)[0]
    ind = np.random.choice(indices, 1)[0]
    y, x = ind // w, ind % w
    theta = np.pi/2
    X = np.array([[x, y, theta]])
    X = np.repeat(X, 2, axis=0)
    X[:, :2] *= 10

    num_beams = 180
    sensor = SensorModel(occupancy_map)
    z_t_star = sensor.ray_casting(X, num_beams=num_beams)

    x0, y0 = X[0, :2]
    angle = np.arange(num_beams) * (np.pi / num_beams)
    angle = theta + angle - np.pi/2
    x1 = x0 + z_t_star * np.cos(angle)
    y1 = y0 - z_t_star * np.sin(angle)

    x0, y0 = x0 / 10, y0 / 10
    x1, y1 = x1 / 10, y1 / 10
    # plot figure
    fig = plt.figure()
    plt.imshow(occupancy_map)
    plt.scatter(x0, y0, c='red')
    plt.scatter(x1, y1, c='yellow')
    print(f'(x0, y0): ({x0}, {y0}), (x1, y1): ({x1}, {y1})')
    # plt.plot((x0, x1), (y0, y1), color='yellow')

    plt.show()
    print(z_t_star)


if __name__ == '__main__':
    test_raycasting_vectorize()