from __future__ import division

import numpy as np


class UniformNet:
    def __init__(self, data_proc, distance):
        self.data = data_proc.data
        self.mins = np.amin(data_proc.data, axis=0) # min values for consecutive dimensions: 0, 1..
        self.maxes = np.amax(data_proc.data, axis=0)
        self.distance = distance

    def __build(self, points, dim):
        if len(points.shape) > 2:
            for i in range(points.shape[dim]):
                points[i] = self.__build(points[i], dim+1)
        else:
            val = self.mins[dim]
            for i in range(len(points)):
                points[i] = np.array([val, dim])
                val += self.distance
        return np.copy(points)

    def build_and_sample(self):
        # build a net:
        points_num = []
        dims = len(self.data.shape)
        for dim in (range(dims)):
            points_num.append(int((self.maxes[dim] - self.mins[dim]) / self.distance) + 1)
        points_num.append(dims)
        points = np.zeros(points_num)
        points = self.__build(points, 0)
        print(points

class ManifoldLandmarker:
    def __init__(self):
        pass

    def __uniform_sampling(self, data_proc):
        #distance between two samples in one chosen dimension
        sample_dist = 10
        shape = data_proc.data.shape
        net = UniformNet(data_proc, 0.5)
        net.build_and_sample()



    def create_knn_graph(self, data_proc):
        self.__uniform_sampling(data_proc)