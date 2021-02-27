import numpy as np
import matplotlib.pyplot as plt

from map_reader import MapReader
from sensor_model import SensorModel, MIN_PROBABILITY


def test_hyperparameter():
    np.random.seed(10008)
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    # generate a random particle, then sent out beams from that location
    h, w = occupancy_map.shape
    indices = np.where(occupancy_map.flatten() == 0)[0]
    ind = np.random.choice(indices, 1)[0]
    y, x = ind // w, ind % w
    theta = -np.pi / 2
    angle = np.pi * (40 / 180)

    sensor = SensorModel(occupancy_map)
    z_t_star = sensor.ray_casting_one_direction(np.array([x, y, theta]), occupancy_map, angle)
    z_t_star *= 10
    print(z_t_star)

    z = np.arange(sensor._max_range).astype(np.float)
    p_hit, p_short, p_max, p_rand = sensor.estimate_density(z, z_t_star)
    plot(1, p_hit)
    plot(2, p_short)
    plot(3, p_max)
    plot(4, p_rand)

    w_hit = 99 / 2 / 2.5 / 4  # 1.
    w_short = 2 * 198 / 4 / 2.5 / 4  # 1
    w_max = 49 / 2.5 / 4  # 0.5
    w_rand = 990 / 4  # 5

    # self._z_hit = 99 / 2 / 2.5 / 4  # 1.
    # self._z_short = 198 / 4 // 2.5 / 4  # 1
    # self._z_max = 49 / 2.5 / 4  # 0.5
    # self._z_rand = 990 / 4  # 5

    # w_hit = 1.
    # w_short = 0.1
    # w_max = 0.5
    # w_rand = 10
    p = w_hit * p_hit + w_short * p_short + w_max * p_max + w_rand * p_rand
    plot(5, p)
    plt.show()


def plot(figid, distribution):
    fig = plt.figure(figid)
    plt.bar(np.arange(len(distribution)), distribution)
    # plt.ylim(0, 0.2)
    plt.ylim(0, 1)

if __name__ == '__main__':
    test_hyperparameter()